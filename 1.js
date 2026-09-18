/**
 * DENIUS TEMPORAL LEARNING CORE v2
 * -------------------------------------------------------------
 * 2-file Sunwin Tài/Xỉu analyzer.
 *
 * Design goals:
 * - Không dùng luật "thua N lần thì đảo cửa".
 * - Không gán confidence cố định 65/75/88%.
 * - Học walk-forward: dự đoán trước -> nhận nhãn thật -> cập nhật.
 * - Stacking: Logistic + MLP + Context Bayesian + KNN -> Meta learner.
 * - Tự đánh giá bằng Accuracy / LogLoss / Brier trên dữ liệu out-of-sample.
 * - Rebuild được từ lịch sử API nếu state runtime bị mất.
 * - Lưu state để tiếp tục học giữa các lần restart khi filesystem còn.
 *
 * Lưu ý: Tài/Xỉu là một quá trình có thể ngẫu nhiên. Hệ thống này
 * không thể đảm bảo thắng hay dự đoán chắc chắn.
 */

'use strict';

const express = require('express');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = Number(process.env.PORT || 3000);
const API_URL = process.env.API_URL || 'https://sunwin-taixiu-dulieu.onrender.com/data';
const POLL_MS = Math.max(5000, Number(process.env.POLL_MS || 15000));
const HISTORY_LIMIT = Math.max(300, Number(process.env.HISTORY_LIMIT || 2000));
const KNN_LIMIT = Math.max(200, Number(process.env.KNN_LIMIT || 800));
const WARMUP = Math.max(30, Number(process.env.WARMUP || 40));
const STATE_FILE = process.env.STATE_FILE || path.join(process.cwd(), 'denius_brain_state.json');

const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, x));
const sigmoid = x => {
    if (x >= 30) return 1;
    if (x <= -30) return 0;
    return 1 / (1 + Math.exp(-x));
};
const logit = p => Math.log(clamp(p, 1e-6, 1 - 1e-6) / clamp(1 - p, 1e-6, 1 - 1e-6));
const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
const variance = xs => {
    if (xs.length < 2) return 0;
    const m = mean(xs);
    return mean(xs.map(x => (x - m) ** 2));
};
const std = xs => Math.sqrt(Math.max(0, variance(xs)));
const safe = (x, fallback = 0) => Number.isFinite(Number(x)) ? Number(x) : fallback;
const isoVN = () => new Date().toLocaleString('sv-SE', { timeZone: 'Asia/Ho_Chi_Minh' }).replace(' ', 'T');

function normSide(v) {
    const s = String(v ?? '').toUpperCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    if (s.includes('TAI')) return 'TAI';
    if (s.includes('XIU')) return 'XIU';
    return null;
}

function normalizeRound(raw) {
    if (!raw || typeof raw !== 'object') return null;
    // Accept both ASCII aliases and the Vietnamese fields used by the source API.
    const phienRaw = raw.phien ?? raw['phiên'] ?? raw.session;
    if (phienRaw == null) return null;
    const phien = Number(phienRaw);
    if (!Number.isFinite(phien)) return null;
    const d1 = Number(raw.d1 ?? raw['d1'] ?? raw.xuc_xac_1 ?? raw.dice1 ?? 0);
    const d2 = Number(raw.d2 ?? raw['d2'] ?? raw.xuc_xac_2 ?? raw.dice2 ?? 0);
    const d3 = Number(raw.d3 ?? raw['d3'] ?? raw.xuc_xac_3 ?? raw.dice3 ?? 0);
    const totalRaw = raw.tong ?? raw['tổng'] ?? raw.total ?? (d1 && d2 && d3 ? d1 + d2 + d3 : null);
    const tong = totalRaw == null ? null : Number(totalRaw);
    const ket_qua = normSide(raw.ket_qua ?? raw.ketqua ?? raw['kết quả'] ?? raw.result);
    if (!ket_qua) return null;
    return {
        phien,
        d1: Number.isFinite(d1) ? d1 : 0,
        d2: Number.isFinite(d2) ? d2 : 0,
        d3: Number.isFinite(d3) ? d3 : 0,
        tong: Number.isFinite(tong) ? tong : null,
        ket_qua,
        thoi_gian: raw.thoi_gian ?? raw.updatedAt ?? raw.time ?? null
    };
}

function normalizePayload(raw) {
    const candidates = [
        raw?.data,
        raw?.history,
        raw?.records,
        raw?.result,
        raw?.items,
        raw?.data?.data,
        raw?.result?.data,
        raw
    ];
    const arr = candidates.find(Array.isArray) || [];
    const map = new Map();
    for (const item of arr) {
        const r = normalizeRound(item);
        if (r) map.set(r.phien, r);
    }
    return [...map.values()]
        .sort((a, b) => a.phien - b.phien)
        .slice(-HISTORY_LIMIT);
}

function oneHotSide(side) {
    return side === 'TAI' ? [1, 0] : side === 'XIU' ? [0, 1] : [0.5, 0.5];
}

/**
 * Feature extraction is causal: only `past` is used.
 * No current label leaks into the feature vector.
 */
function extractFeatures(past) {
    const f = [];
    const arr = past;
    const sides = arr.map(r => r.ket_qua);
    const totals = arr.map(r => r.tong).filter(Number.isFinite);
    const last = arr[arr.length - 1] || {};
    const prior = arr[arr.length - 2] || {};

    // 1) Recent categorical context: fixed one-hot history up to 10 rounds.
    for (let i = 1; i <= 10; i++) {
        const s = sides[sides.length - i];
        f.push(...oneHotSide(s));
    }

    // 2) Long/short outcome rates.
    for (const n of [5, 10, 20, 50, 100]) {
        const w = sides.slice(-n);
        const pTai = w.length ? w.filter(x => x === 'TAI').length / w.length : 0.5;
        f.push(pTai, 1 - pTai);
    }

    // 3) Streak / transition structure.
    let streak = 0;
    if (sides.length) {
        const s0 = sides[sides.length - 1];
        for (let i = sides.length - 1; i >= 0 && sides[i] === s0; i--) streak++;
        f.push(streak / 12, s0 === 'TAI' ? 1 : 0, s0 === 'XIU' ? 1 : 0);
    } else {
        f.push(0, 0.5, 0.5);
    }

    for (const n of [10, 30, 100]) {
        const w = sides.slice(-n);
        let switches = 0;
        for (let i = 1; i < w.length; i++) if (w[i] !== w[i - 1]) switches++;
        f.push(w.length > 1 ? switches / (w.length - 1) : 0.5);
    }

    // 4) Total statistics.
    for (const n of [5, 10, 20, 50, 100]) {
        const w = totals.slice(-n);
        const m = mean(w);
        const sd = std(w);
        f.push(
            clamp((m - 3) / 15),
            clamp(sd / 6),
            clamp((Math.min(...(w.length ? w : [10])) - 3) / 15),
            clamp((Math.max(...(w.length ? w : [10])) - 3) / 15)
        );
    }

    // 5) Recent total trajectory and deltas.
    const t = totals.slice(-12);
    for (let i = 1; i < t.length; i++) f.push(clamp((t[i] - t[i - 1]) / 15, -1, 1));
    while (f.length < 20 + 10 + 3 + 3 + 20 + 11) f.push(0.5);
    f.push(clamp(((safe(last.tong, 10) - 10) / 7), -1, 1));
    f.push(clamp(((safe(last.d1, 3) - 3) / 2), -1, 1));
    f.push(clamp(((safe(last.d2, 3) - 3) / 2), -1, 1));
    f.push(clamp(((safe(last.d3, 3) - 3) / 2), -1, 1));
    f.push(last.d1 === last.d2 ? 1 : 0, last.d2 === last.d3 ? 1 : 0, last.d1 === last.d3 ? 1 : 0);
    f.push(last.d1 === 1 || last.d2 === 1 || last.d3 === 1 ? 1 : 0);
    f.push(last.d1 === 6 || last.d2 === 6 || last.d3 === 6 ? 1 : 0);
    f.push((safe(last.tong, 10) % 2) / 1);
    f.push(clamp((safe(last.tong, 10) - safe(prior.tong, 10)) / 15, -1, 1));

    // Replace non-finite values, preserve all signal dimensions.
    return f.map(x => Number.isFinite(x) ? x : 0.5);
}

const FEATURE_COUNT = extractFeatures([
    { ket_qua: 'TAI', tong: 11, d1: 3, d2: 4, d3: 4 },
    { ket_qua: 'XIU', tong: 9, d1: 3, d2: 3, d3: 3 },
    { ket_qua: 'TAI', tong: 12, d1: 6, d2: 2, d3: 4 }
]).length;

class OnlineLogistic {
    constructor(n = FEATURE_COUNT, lr = 0.015, l2 = 0.0005) {
        this.n = n;
        this.lr = lr;
        this.l2 = l2;
        this.w = Array(n).fill(0);
        this.b = 0;
        this.m = Array(n).fill(0);
        this.v = Array(n).fill(0);
        this.mb = 0;
        this.vb = 0;
        this.t = 0;
        this.samples = 0;
    }
    predict(x) {
        let z = this.b;
        for (let i = 0; i < this.n; i++) z += (x[i] || 0) * this.w[i];
        return sigmoid(z);
    }
    update(x, y) {
        const p = this.predict(x);
        const g = p - y;
        this.t++;
        const b1 = 0.9, b2 = 0.999, eps = 1e-8;
        for (let i = 0; i < this.n; i++) {
            const gi = g * (x[i] || 0) + this.l2 * this.w[i];
            this.m[i] = b1 * this.m[i] + (1 - b1) * gi;
            this.v[i] = b2 * this.v[i] + (1 - b2) * gi * gi;
            const mh = this.m[i] / (1 - Math.pow(b1, this.t));
            const vh = this.v[i] / (1 - Math.pow(b2, this.t));
            this.w[i] -= this.lr * mh / (Math.sqrt(vh) + eps);
        }
        this.mb = b1 * this.mb + (1 - b1) * g;
        this.vb = b2 * this.vb + (1 - b2) * g * g;
        const mbh = this.mb / (1 - Math.pow(b1, this.t));
        const vbh = this.vb / (1 - Math.pow(b2, this.t));
        this.b -= this.lr * mbh / (Math.sqrt(vbh) + eps);
        this.samples++;
        return p;
    }
    export() {
        return { n: this.n, lr: this.lr, l2: this.l2, w: this.w, b: this.b, m: this.m, v: this.v, mb: this.mb, vb: this.vb, t: this.t, samples: this.samples };
    }
    import(s) {
        if (!s || !Array.isArray(s.w) || s.w.length !== this.n) return false;
        Object.assign(this, s);
        return true;
    }
}

class OnlineMLP {
    constructor(input = FEATURE_COUNT, h1 = 20, h2 = 10, lr = 0.003) {
        this.input = input;
        this.h1 = h1;
        this.h2 = h2;
        this.lr = lr;
        this.t = 0;
        this.samples = 0;
        this.W1 = this.matrix(h1, input, 0.12);
        this.b1 = Array(h1).fill(0);
        this.W2 = this.matrix(h2, h1, 0.12);
        this.b2 = Array(h2).fill(0);
        this.W3 = Array(h2).fill(0).map(() => this.randWeight());
        this.b3 = 0;
        this.opt = {
            mW1: this.zerosLike(this.W1), vW1: this.zerosLike(this.W1),
            mb1: Array(h1).fill(0), vb1: Array(h1).fill(0),
            mW2: this.zerosLike(this.W2), vW2: this.zerosLike(this.W2),
            mb2: Array(h2).fill(0), vb2: Array(h2).fill(0),
            mW3: Array(h2).fill(0), vW3: Array(h2).fill(0), mb3: 0, vb3: 0
        };
    }
    randWeight() { return (Math.random() * 2 - 1) * 0.15; }
    matrix(rows, cols) { return Array(rows).fill(0).map(() => Array(cols).fill(0).map(() => this.randWeight())); }
    zerosLike(m) { return m.map(row => row.map(() => 0)); }
    forward(x) {
        const z1 = this.W1.map((row, i) => row.reduce((s, w, j) => s + w * (x[j] || 0), this.b1[i]));
        const a1 = z1.map(v => Math.tanh(v));
        const z2 = this.W2.map((row, i) => row.reduce((s, w, j) => s + w * a1[j], this.b2[i]));
        const a2 = z2.map(v => Math.tanh(v));
        const z3 = this.b3 + this.W3.reduce((s, w, i) => s + w * a2[i], 0);
        return { p: sigmoid(z3), z1, a1, z2, a2, z3 };
    }
    predict(x) { return this.forward(x).p; }
    update(x, y) {
        const { p, a1, a2 } = this.forward(x);
        const dz3 = p - y;
        const da2 = this.W3.map(w => dz3 * w);
        const dz2 = da2.map((v, i) => v * (1 - a2[i] * a2[i]));
        const da1 = Array(this.h1).fill(0);
        for (let i = 0; i < this.h2; i++) for (let j = 0; j < this.h1; j++) da1[j] += dz2[i] * this.W2[i][j];
        const dz1 = da1.map((v, i) => v * (1 - a1[i] * a1[i]));

        this.t++;
        this.adamVector(this.W3, dz3, a2, 'W3');
        this.adamScalar('b3', dz3, 'mb3', 'vb3');
        for (let i = 0; i < this.h2; i++) {
            this.adamRow(this.W2[i], dz2[i], a1, i, 'W2');
            this.adamVectorIndex('b2', i, dz2[i]);
        }
        for (let i = 0; i < this.h1; i++) {
            this.adamRow(this.W1[i], dz1[i], x, i, 'W1');
            this.adamVectorIndex('b1', i, dz1[i]);
        }
        this.samples++;
        return p;
    }
    adamVector(vec, scalar, source, name) {
        const m = this.opt['m' + name], v = this.opt['v' + name];
        for (let i = 0; i < vec.length; i++) {
            const g = scalar * source[i];
            m[i] = 0.9 * m[i] + 0.1 * g;
            v[i] = 0.999 * v[i] + 0.001 * g * g;
            const mh = m[i] / (1 - Math.pow(0.9, this.t));
            const vh = v[i] / (1 - Math.pow(0.999, this.t));
            vec[i] -= this.lr * mh / (Math.sqrt(vh) + 1e-8);
        }
    }
    adamRow(row, scalar, source, rowIndex, name) {
        const m = this.opt['m' + name][rowIndex], v = this.opt['v' + name][rowIndex];
        for (let j = 0; j < row.length; j++) {
            const g = scalar * source[j];
            m[j] = 0.9 * m[j] + 0.1 * g;
            v[j] = 0.999 * v[j] + 0.001 * g * g;
            const mh = m[j] / (1 - Math.pow(0.9, this.t));
            const vh = v[j] / (1 - Math.pow(0.999, this.t));
            row[j] -= this.lr * mh / (Math.sqrt(vh) + 1e-8);
        }
    }
    adamVectorIndex(group, i, g) {
        const mKey = 'm' + group, vKey = 'v' + group;
        let m = this.opt[mKey][i], v = this.opt[vKey][i];
        m = 0.9 * m + 0.1 * g;
        v = 0.999 * v + 0.001 * g * g;
        const mh = m / (1 - Math.pow(0.9, this.t));
        const vh = v / (1 - Math.pow(0.999, this.t));
        this.opt[mKey][i] = m;
        this.opt[vKey][i] = v;
        const biasKey = group === 'b1' ? 'b1' : 'b2';
        this[biasKey][i] -= this.lr * mh / (Math.sqrt(vh) + 1e-8);
    }
    adamScalar(prop, g, mProp, vProp) {
        let m = this.opt[mProp], v = this.opt[vProp];
        m = 0.9 * m + 0.1 * g;
        v = 0.999 * v + 0.001 * g * g;
        const mh = m / (1 - Math.pow(0.9, this.t));
        const vh = v / (1 - Math.pow(0.999, this.t));
        this.opt[mProp] = m;
        this.opt[vProp] = v;
        this[prop] -= this.lr * mh / (Math.sqrt(vh) + 1e-8);
    }
    export() { return { input: this.input, h1: this.h1, h2: this.h2, lr: this.lr, t: this.t, samples: this.samples, W1: this.W1, b1: this.b1, W2: this.W2, b2: this.b2, W3: this.W3, b3: this.b3, opt: this.opt }; }
    import(s) {
        if (!s || s.input !== this.input || s.h1 !== this.h1 || s.h2 !== this.h2) return false;
        Object.assign(this, s);
        return true;
    }
}

class ContextBayes {
    constructor(maxOrder = 6, alpha = 1) {
        this.maxOrder = maxOrder;
        this.alpha = alpha;
        this.counts = {};
        this.samples = 0;
    }
    key(sides) { return sides.join(''); }
    predict(past) {
        const sides = past.map(r => r.ket_qua);
        let best = 0.5;
        for (let order = this.maxOrder; order >= 1; order--) {
            if (sides.length < order) continue;
            const k = this.key(sides.slice(-order));
            const c = this.counts[k];
            if (!c || c.n < Math.max(3, order)) continue;
            best = (c.tai + this.alpha) / (c.n + 2 * this.alpha);
            break;
        }
        return best;
    }
    update(past, y) {
        const sides = past.map(r => r.ket_qua);
        for (let order = 1; order <= this.maxOrder; order++) {
            if (sides.length < order) continue;
            const k = this.key(sides.slice(-order));
            if (!this.counts[k]) this.counts[k] = { tai: 0, xiu: 0, n: 0 };
            this.counts[k].n++;
            if (y === 1) this.counts[k].tai++;
            else this.counts[k].xiu++;
        }
        this.samples++;
    }
    export() { return { maxOrder: this.maxOrder, alpha: this.alpha, counts: this.counts, samples: this.samples }; }
    import(s) {
        if (!s || !s.counts) return false;
        Object.assign(this, s);
        return true;
    }
}

class PatternKNN {
    constructor(limit = KNN_LIMIT, k = 24) {
        this.limit = limit;
        this.k = k;
        this.samples = [];
    }
    predict(x) {
        if (this.samples.length < 12) return 0.5;
        const scored = [];
        for (const s of this.samples) {
            let d = 0;
            for (let i = 0; i < x.length; i++) {
                const z = (x[i] || 0) - (s.x[i] || 0);
                d += z * z;
            }
            scored.push({ d: Math.sqrt(d), y: s.y });
        }
        scored.sort((a, b) => a.d - b.d);
        const top = scored.slice(0, this.k);
        let num = 0, den = 0;
        for (const s of top) {
            const w = 1 / (0.15 + s.d);
            num += s.y * w;
            den += w;
        }
        return den ? clamp(num / den, 0.01, 0.99) : 0.5;
    }
    update(x, y) {
        this.samples.push({ x: x.slice(), y });
        if (this.samples.length > this.limit) this.samples.shift();
    }
    export() { return { limit: this.limit, k: this.k, samples: this.samples }; }
    import(s) {
        if (!s || !Array.isArray(s.samples)) return false;
        Object.assign(this, s);
        return true;
    }
}

class TemporalBrain {
    constructor() { this.reset(); }
    reset() {
        this.logistic = new OnlineLogistic();
        this.mlp = new OnlineMLP();
        this.context = new ContextBayes();
        this.knn = new PatternKNN();
        this.meta = new OnlineLogistic(4, 0.02, 0.001);
        this.history = [];
        this.seen = new Set();
        this.metrics = { predictions: 0, correct: 0, logLoss: [], brier: [], recent: [] };
        this.trainingSamples = 0;
        this.lastTrainedSession = null;
        this.lastBuild = null;
        this.lastPrediction = null;
        this.predictionLog = [];
        this.events = [];
    }
    y(round) { return round.ket_qua === 'TAI' ? 1 : 0; }
    basePredictions(past, x) {
        return {
            logistic: this.logistic.predict(x),
            mlp: this.mlp.predict(x),
            context: this.context.predict(past),
            knn: this.knn.predict(x)
        };
    }
    metaVector(b) {
        return [b.logistic, b.mlp, b.context, b.knn];
    }
    ensemble(past, x) {
        const b = this.basePredictions(past, x);
        const mv = this.metaVector(b);
        let p = this.meta.samples < 25 ? mean(Object.values(b)) : this.meta.predict(mv);
        const spread = std(Object.values(b));
        // Evidence quality comes from disagreement, not a hard-coded confidence.
        const uncertainty = clamp(0.5 + spread * 0.9 - Math.abs(p - 0.5) * 0.5, 0, 0.5);
        return { p: clamp(p, 0.001, 0.999), base: b, spread, uncertainty };
    }
    observePrediction(p, actual) {
        const y = actual === 'TAI' ? 1 : 0;
        const pp = clamp(p, 1e-6, 1 - 1e-6);
        const ll = -(y * Math.log(pp) + (1 - y) * Math.log(1 - pp));
        const br = (pp - y) ** 2;
        const ok = (pp >= 0.5 ? 1 : 0) === y;
        this.metrics.predictions++;
        if (ok) this.metrics.correct++;
        this.metrics.logLoss.push(ll);
        this.metrics.brier.push(br);
        if (this.metrics.logLoss.length > 300) this.metrics.logLoss.shift();
        if (this.metrics.brier.length > 300) this.metrics.brier.shift();
        this.metrics.recent.push(ok ? 1 : 0);
        if (this.metrics.recent.length > 100) this.metrics.recent.shift();
    }
    learnOne(round, recordShadow = true) {
        const past = this.history;
        if (past.length < WARMUP) {
            this.history.push(round);
            this.seen.add(round.phien);
            this.lastTrainedSession = round.phien;
            return null;
        }
        const x = extractFeatures(past);
        const b = this.basePredictions(past, x);
        const metaX = this.metaVector(b);
        const pBefore = this.meta.samples < 25 ? mean(Object.values(b)) : this.meta.predict(metaX);
        if (recordShadow) this.observePrediction(pBefore, round.ket_qua);

        this.meta.update(metaX, this.y(round));
        this.logistic.update(x, this.y(round));
        this.mlp.update(x, this.y(round));
        this.context.update(past, this.y(round));
        this.knn.update(x, this.y(round));
        this.trainingSamples++;
        this.history.push(round);
        this.seen.add(round.phien);
        this.lastTrainedSession = round.phien;
        return { p: pBefore, base: b, actual: round.ket_qua };
    }
    rebuild(data) {
        const clean = data.slice().sort((a, b) => a.phien - b.phien).slice(-HISTORY_LIMIT);
        this.reset();
        this.lastBuild = isoVN();
        const shadow = [];
        for (const r of clean) {
            const res = this.learnOne(r, true);
            if (res) shadow.push({ phien: r.phien, p: res.p, actual: r.ket_qua });
        }
        this.events.unshift({ time: isoVN(), type: 'REBUILD', samples: this.trainingSamples, rows: clean.length });
        this.events = this.events.slice(0, 20);
        return shadow.length;
    }
    ingest(data) {
        const clean = data.slice().sort((a, b) => a.phien - b.phien);
        if (!this.history.length) {
            this.rebuild(clean);
            return { rebuilt: true, learned: this.trainingSamples };
        }
        const maxKnown = this.history[this.history.length - 1]?.phien ?? 0;
        const incoming = clean.filter(r => r.phien > maxKnown);
        let learned = 0;
        for (const r of incoming) {
            const prior = this.lastPrediction && this.lastPrediction.phienDuDoan === r.phien ? this.lastPrediction : null;
            const res = this.learnOne(r, true);
            if (prior && res) {
                const actual = r.ket_qua;
                const ok = prior.pred === actual;
                this.predictionLog.unshift({
                    phien: r.phien,
                    predict: prior.pred,
                    actual,
                    probability: prior.p,
                    confidence: Math.round(Math.max(prior.p, 1 - prior.p) * 1000) / 10,
                    uncertainty: Math.round(prior.uncertainty * 1000) / 10,
                    correct: ok,
                    source: prior.type,
                    time: r.thoi_gian || isoVN()
                });
                this.predictionLog = this.predictionLog.slice(0, 120);
            }
            learned++;
        }
        if (incoming.length) this.events.unshift({ time: isoVN(), type: 'ONLINE_UPDATE', learned: incoming.length, latest: incoming[incoming.length - 1].phien });
        this.events = this.events.slice(0, 20);
        if (this.history.length > HISTORY_LIMIT) {
            this.history = this.history.slice(-HISTORY_LIMIT);
        }
        return { rebuilt: false, learned };
    }
    predictNext() {
        if (this.history.length < WARMUP) return null;
        const x = extractFeatures(this.history);
        const e = this.ensemble(this.history, x);
        const pred = e.p >= 0.5 ? 'TAI' : 'XIU';
        const confidence = Math.round(Math.max(e.p, 1 - e.p) * 1000) / 10;
        const uncertainty = Math.round(e.uncertainty * 1000) / 10;
        const next = (this.history[this.history.length - 1]?.phien || 0) + 1;
        const source = this.meta.samples < 25 ? 'BASE-COLDSTART' : 'META-STACK';
        this.lastPrediction = {
            phienDuDoan: next,
            pred,
            display: pred === 'TAI' ? 'Tài' : 'Xỉu',
            p: Number(e.p.toFixed(5)),
            confidence,
            uncertainty,
            type: source,
            base: Object.fromEntries(Object.entries(e.base).map(([k, v]) => [k, Number(v.toFixed(5))])),
            modelSamples: this.trainingSamples,
            timestamp: isoVN()
        };
        return this.lastPrediction;
    }
    summary() {
        const n = this.metrics.predictions;
        return {
            predictions: n,
            correct: this.metrics.correct,
            accuracy: n ? this.metrics.correct / n : 0,
            logLoss: mean(this.metrics.logLoss),
            brier: mean(this.metrics.brier),
            recentAccuracy: this.metrics.recent.length ? mean(this.metrics.recent) : 0
        };
    }
    exportState() {
        return {
            version: 2,
            savedAt: isoVN(),
            featureCount: FEATURE_COUNT,
            history: this.history,
            metrics: this.metrics,
            trainingSamples: this.trainingSamples,
            lastTrainedSession: this.lastTrainedSession,
            lastBuild: this.lastBuild,
            lastPrediction: this.lastPrediction,
            predictionLog: this.predictionLog,
            events: this.events,
            logistic: this.logistic.export(),
            mlp: this.mlp.export(),
            context: this.context.export(),
            knn: this.knn.export(),
            meta: this.meta.export()
        };
    }
    importState(s) {
        if (!s || s.version !== 2 || s.featureCount !== FEATURE_COUNT || !Array.isArray(s.history)) return false;
        this.history = s.history;
        this.metrics = s.metrics || this.metrics;
        this.trainingSamples = s.trainingSamples || 0;
        this.lastTrainedSession = s.lastTrainedSession || null;
        this.lastBuild = s.lastBuild || null;
        this.lastPrediction = s.lastPrediction || null;
        this.predictionLog = Array.isArray(s.predictionLog) ? s.predictionLog : [];
        this.events = Array.isArray(s.events) ? s.events : [];
        this.seen = new Set(this.history.map(r => r.phien));
        const ok = [
            this.logistic.import(s.logistic),
            this.mlp.import(s.mlp),
            this.context.import(s.context),
            this.knn.import(s.knn),
            this.meta.import(s.meta)
        ].every(Boolean);
        return ok;
    }
}

const brain = new TemporalBrain();
const appStats = {
    startedAt: isoVN(),
    lastFetch: null,
    lastApiRows: 0,
    apiErrors: 0,
    lastError: null,
    syncs: 0
};

function loadState() {
    try {
        if (!fs.existsSync(STATE_FILE)) return false;
        const raw = JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
        const ok = brain.importState(raw);
        if (ok) console.log(`[STATE] Loaded ${brain.trainingSamples} learned samples.`);
        return ok;
    } catch (err) {
        console.error('[STATE] Load failed:', err.message);
        return false;
    }
}

let saveTimer = null;
function saveStateSoon() {
    if (saveTimer) return;
    saveTimer = setTimeout(() => {
        saveTimer = null;
        try {
            fs.writeFileSync(STATE_FILE, JSON.stringify(brain.exportState()), 'utf8');
        } catch (err) {
            console.error('[STATE] Save failed:', err.message);
        }
    }, 750);
}

async function fetchHistory() {
    try {
        const joiner = API_URL.includes('?') ? '&' : '?';
        const requestUrl = API_URL + joiner + 't=' + Date.now();
        const res = await axios.get(requestUrl, {
            timeout: 12000,
            headers: {
                'User-Agent': 'DENIUS-Temporal-Core/2.1',
                'Accept': 'application/json'
            },
            validateStatus: s => s >= 200 && s < 400
        });
        const data = normalizePayload(res.data);
        appStats.lastFetch = isoVN();
        appStats.lastApiRows = data.length;
        appStats.apiErrors = 0;
        appStats.lastError = null;
        appStats.syncs++;
        if (!data.length) {
            const shape = Array.isArray(res.data)
                ? 'array'
                : (res.data && typeof res.data === 'object'
                    ? 'object keys=' + Object.keys(res.data).slice(0, 12).join(',')
                    : typeof res.data);
            throw new Error('API trả payload nhưng không map được phiên. ' + shape);
        }
        const result = brain.ingest(data);
        const latest = data[data.length - 1];
        // Prediction always uses only history that is already complete.
        if (!brain.lastPrediction || brain.lastPrediction.phienDuDoan !== latest.phien + 1) {
            brain.predictNext();
        } else {
            // Recalculate with latest model after online learning.
            brain.predictNext();
        }
        saveStateSoon();
        console.log(`[SYNC] rows=${data.length} learned=${result.learned} latest=#${latest.phien} next=${brain.lastPrediction?.pred || '-'} p=${brain.lastPrediction?.p ?? '-'} acc=${(brain.summary().accuracy * 100).toFixed(1)}%`);
    } catch (err) {
        appStats.apiErrors++;
        appStats.lastError = err.message;
        console.error('[API] Fetch error:', err.message);
    }
}

// ------------------------- DASHBOARD -------------------------
const html = `<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DENIUS Temporal Learning Core</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#050711;--panel:rgba(12,18,34,.88);--line:rgba(112,180,255,.16);--cyan:#43d9ff;--violet:#9b6bff;--green:#42e6a4;--red:#ff5d7a;--yellow:#ffd45c;--text:#eef6ff;--muted:#7f93ac}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 10%,rgba(67,217,255,.08),transparent 28%),radial-gradient(circle at 85% 0%,rgba(155,107,255,.08),transparent 26%),linear-gradient(180deg,#050711,#080b16 65%,#050711);color:var(--text);font-family:Inter,system-ui,sans-serif;min-height:100vh}
.wrap{max-width:1180px;margin:auto;padding:28px 18px 48px}.hero{text-align:center;margin-bottom:22px}.logo{font:800 2rem Orbitron;letter-spacing:2px;background:linear-gradient(90deg,var(--cyan),#fff,var(--violet));-webkit-background-clip:text;color:transparent}.sub{margin-top:7px;color:var(--muted);font-size:.92rem}.live{display:inline-flex;gap:8px;align-items:center;margin-top:13px;border:1px solid var(--line);padding:7px 13px;border-radius:999px;color:var(--cyan);font-size:.78rem}.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 10px var(--green)}
.grid{display:grid;grid-template-columns:1.3fr 1fr;gap:16px}.card{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 16px 55px rgba(0,0,0,.24);backdrop-filter:blur(15px)}.title{font:700 .74rem Orbitron;letter-spacing:1.4px;color:var(--cyan);text-transform:uppercase;margin-bottom:16px}.pred{font:800 3.25rem Orbitron;line-height:1;margin:8px 0}.tai{color:var(--green);text-shadow:0 0 28px rgba(66,230,164,.2)}.xiu{color:var(--red);text-shadow:0 0 28px rgba(255,93,122,.2)}.prob{font:700 1.5rem Orbitron;color:var(--yellow)}.small{color:var(--muted);font-size:.86rem;line-height:1.65}.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.chip{border:1px solid var(--line);padding:6px 10px;border-radius:10px;font-size:.76rem;color:#b9cbe0;background:rgba(255,255,255,.02)}
.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.stat{padding:14px;border-radius:13px;border:1px solid var(--line);background:rgba(255,255,255,.018)}.num{font:700 1.35rem Orbitron}.lab{margin-top:4px;color:var(--muted);font-size:.72rem}.ok{color:var(--green)}.bad{color:var(--red)}.cy{color:var(--cyan)}.yl{color:var(--yellow)}
.section{margin-top:16px}.bars{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.bar{height:100px;border:1px solid var(--line);border-radius:12px;padding:10px;display:flex;flex-direction:column;justify-content:flex-end}.fill{height:6px;border-radius:99px;background:linear-gradient(90deg,var(--cyan),var(--violet));margin-bottom:7px}.bt{font-size:.72rem;color:var(--muted)}.bv{font:600 .8rem Orbitron}
.table{overflow:auto}.table table{width:100%;border-collapse:collapse;font-size:.8rem}.table th{font:600 .68rem Orbitron;color:var(--cyan);text-align:left;padding:9px;border-bottom:1px solid var(--line)}.table td{padding:9px;border-bottom:1px solid rgba(255,255,255,.05)}.pill{padding:4px 8px;border-radius:7px;font-size:.68rem;border:1px solid var(--line)}.p1{color:var(--green)}.p0{color:var(--red)}
.footer{margin-top:18px;color:var(--muted);text-align:center;font-size:.75rem;line-height:1.7}.wide{grid-column:1/-1}@media(max-width:860px){.grid{grid-template-columns:1fr}.bars{grid-template-columns:repeat(2,1fr)}.pred{font-size:2.6rem}}
</style>
</head><body><div class="wrap">
<div class="hero"><div class="logo">DENIUS · TEMPORAL CORE</div><div class="sub">Online Learning · Walk-Forward Validation · Adaptive Stacking</div><div class="live"><span class="dot"></span><span id="status">NEURAL ENGINE SYNCING</span></div></div>
<div class="grid">
<div class="card"><div class="title">Next Session</div><div id="pred" class="pred">---</div><div id="prob" class="prob">--.-%</div><div class="small">Phiên <b id="session">#---</b> · Model <b id="type">---</b><br>Uncertainty <b id="unc">--.-%</b> · Samples <b id="samples">0</b></div><div class="chips"><span class="chip">causal features</span><span class="chip">out-of-sample</span><span class="chip">online update</span><span class="chip">no manual flip</span></div></div>
<div class="card"><div class="title">Learning Metrics</div><div class="stats"><div class="stat"><div class="num cy" id="acc">0.0%</div><div class="lab">Walk-forward accuracy</div></div><div class="stat"><div class="num yl" id="ll">0.000</div><div class="lab">Log loss</div></div><div class="stat"><div class="num" id="br">0.000</div><div class="lab">Brier score</div></div><div class="stat"><div class="num ok" id="ra">0.0%</div><div class="lab">Recent 100 accuracy</div></div></div><div class="small" style="margin-top:14px">API rows: <b id="rows">0</b> · Syncs: <b id="syncs">0</b> · Last fetch: <b id="fetch">-</b></div></div>
<div class="card wide"><div class="title">Model Agreement</div><div id="bars" class="bars"></div><div class="small" style="margin-top:12px">Các base model không được gán confidence thủ công. Meta learner học từ dự đoán out-of-sample của chính các model này.</div></div>
<div class="card wide"><div class="title">Prediction History</div><div class="table"><table><thead><tr><th>Phiên</th><th>Dự đoán</th><th>Thực tế</th><th>P(Tài)</th><th>Conf</th><th>U</th><th>KQ</th></tr></thead><tbody id="log"><tr><td colspan="7">Waiting for labeled predictions…</td></tr></tbody></table></div></div>
<div class="card wide"><div class="title">Engine State</div><div class="small" id="state">Initializing…</div></div>
</div><div class="footer">Tài/Xỉu có thể ngẫu nhiên. Đây là hệ phân tích/học thống kê, không phải đảm bảo kết quả.<br>Auto sync ${Math.round(POLL_MS/1000)}s · Runtime state: ${path.basename(STATE_FILE)}</div>
</div>
<script>
const fmt=(x,d=3)=>Number.isFinite(Number(x))?Number(x).toFixed(d):'0.000';
async function load(){try{const r=await fetch('/api/dashboard',{cache:'no-store'});const d=await r.json();
 const p=d.prediction;if(p){const e=document.getElementById('pred');e.textContent=p.display;e.className='pred '+(p.pred==='TAI'?'tai':'xiu');document.getElementById('prob').textContent=(p.p*100).toFixed(1)+'% P(Tài)';document.getElementById('session').textContent='#'+p.phienDuDoan;document.getElementById('type').textContent=p.type;document.getElementById('unc').textContent=p.uncertainty.toFixed(1)+'%';document.getElementById('samples').textContent=p.modelSamples;}
 const m=d.metrics;document.getElementById('acc').textContent=(m.accuracy*100).toFixed(1)+'%';document.getElementById('ll').textContent=fmt(m.logLoss);document.getElementById('br').textContent=fmt(m.brier);document.getElementById('ra').textContent=(m.recentAccuracy*100).toFixed(1)+'%';document.getElementById('rows').textContent=d.dataCount;document.getElementById('syncs').textContent=d.app.syncs;document.getElementById('fetch').textContent=d.app.lastFetch||'-';document.getElementById('status').textContent=d.app.lastError?'API ERROR':'ONLINE · LEARNING';
 const b=d.prediction?.base||{};document.getElementById('bars').innerHTML=Object.entries(b).map(([k,v])=>{const pct=(v*100);return '<div class="bar"><div class="fill" style="width:'+pct.toFixed(1)+'%"></div><div class="bt">'+k.toUpperCase()+'</div><div class="bv">'+pct.toFixed(1)+'% TÀI</div></div>'}).join('');
 const tb=document.getElementById('log');tb.innerHTML=d.log.length?d.log.map(i=>'<tr><td>#'+i.phien+'</td><td class="'+(i.predict==='TAI'?'p1':'p0')+'">'+(i.predict==='TAI'?'TÀI':'XỈU')+'</td><td class="'+(i.actual==='TAI'?'p1':'p0')+'">'+(i.actual==='TAI'?'TÀI':'XỈU')+'</td><td>'+(i.probability*100).toFixed(1)+'%</td><td>'+i.confidence+'%</td><td>'+i.uncertainty+'%</td><td class="'+(i.correct?'p1':'p0')+'">'+(i.correct?'ĐÚNG':'SAI')+'</td></tr>').join(''):'<tr><td colspan="7">Chưa có prediction đã được chấm.</td></tr>';
 document.getElementById('state').textContent='History '+d.dataCount+' rows · trained '+d.trainingSamples+' samples · last trained #'+(d.lastTrainedSession||'-')+' · feature dim '+d.featureCount+' · rebuild '+(d.lastBuild||'-');
 }catch(e){document.getElementById('status').textContent='DASHBOARD ERROR';}}
load();setInterval(load,8000);
</script></body></html>`;

app.get('/', (req, res) => res.type('html').send(html));
app.get('/health', (req, res) => res.json({ ok: true, service: 'denius-temporal-core', time: isoVN(), dataCount: brain.history.length }));
app.get('/api/dashboard', (req, res) => {
    const summary = brain.summary();
    res.json({
        prediction: brain.lastPrediction,
        metrics: summary,
        log: brain.predictionLog.slice(0, 60),
        error_streak: 0,
        dataCount: brain.history.length,
        trainingSamples: brain.trainingSamples,
        lastTrainedSession: brain.lastTrainedSession,
        lastBuild: brain.lastBuild,
        featureCount: FEATURE_COUNT,
        app: appStats
    });
});
app.get('/api/raw', (req, res) => res.json({ data: brain.history.slice(-100) }));
app.get('/api/model', (req, res) => {
    res.json({
        version: 'DENIUS-TEMPORAL-CORE-v2',
        featureCount: FEATURE_COUNT,
        trainingSamples: brain.trainingSamples,
        models: ['OnlineLogistic(Adam)', 'OnlineMLP(Adam)', 'ContextBayes', 'PatternKNN', 'MetaLogisticStack'],
        walkForward: brain.summary(),
        lastPrediction: brain.lastPrediction
    });
});

app.listen(PORT, async () => {
    console.log('='.repeat(72));
    console.log('DENIUS TEMPORAL LEARNING CORE v2');
    console.log(`Port: ${PORT}`);
    console.log(`API : ${API_URL}`);
    console.log(`Poll: ${POLL_MS}ms`);
    console.log('='.repeat(72));
    const loaded = loadState();
    await fetchHistory();
    if (!loaded && !brain.history.length) await fetchHistory();
    setInterval(fetchHistory, POLL_MS);
});