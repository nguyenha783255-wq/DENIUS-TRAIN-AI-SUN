from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Sequence

WINDOWS = (5, 10, 20, 50, 100)
MAX_HISTORY = 100


def _result_num(r: dict) -> int:
    return 1 if r['result'] == 'Tài' else 0


def _entropy(bits: Sequence[int]) -> float:
    if not bits:
        return 0.0
    p = sum(bits) / len(bits)
    if p <= 0 or p >= 1:
        return 0.0
    return -(p*math.log2(p) + (1-p)*math.log2(1-p))


def _streak(bits: Sequence[int]) -> int:
    if not bits:
        return 0
    last = bits[-1]
    n = 0
    for x in reversed(bits):
        if x != last:
            break
        n += 1
    return n


def _transitions(bits: Sequence[int]) -> float:
    if len(bits) < 2:
        return 0.0
    return sum(a != b for a, b in zip(bits, bits[1:])) / (len(bits)-1)


def make_features(history: Sequence[dict]) -> list[float]:
    # Only history BEFORE the target is allowed to enter this feature vector.
    if not history:
        raise ValueError('history is empty')
    totals = [float(r['total']) for r in history]
    bits = [_result_num(r) for r in history]
    feats: list[float] = []

    # Exact recent sequence: gives trees/logistic models access to local order.
    tail_bits = bits[-12:]
    feats.extend(float(x) for x in tail_bits)
    feats.extend([0.0] * (12-len(tail_bits)))

    # Recent total history.
    tail_totals = totals[-10:]
    feats.extend(tail_totals)
    feats.extend([0.0] * (10-len(tail_totals)))

    for w in WINDOWS:
        b = bits[-w:]
        t = totals[-w:]
        feats += [
            sum(b)/len(b),
            _entropy(b),
            _streak(b)/w,
            _transitions(b),
            mean(t),
            pstdev(t) if len(t) > 1 else 0.0,
            (t[-1] - t[0]) / max(1.0, w-1),
            max(t) - min(t),
        ]

    # Individual dice recency and aggregate statistics.
    for key in ('d1','d2','d3'):
        vals = [float(r[key]) for r in history[-10:]]
        feats.extend(vals)
        feats.extend([0.0] * (10-len(vals)))

    last = history[-1]
    feats.extend([
        float(last['d1']), float(last['d2']), float(last['d3']), float(last['total']),
        float(last['total'] % 2),
        float(sum(x == 6 for x in (last['d1'], last['d2'], last['d3']))),
        float(sum(x == 1 for x in (last['d1'], last['d2'], last['d3']))),
    ])

    # Time spacing, when supplied by source.
    gaps = []
    for a, b in zip(history[-20:], history[-19:]):
        if a.get('updated_at') and b.get('updated_at'):
            try:
                gaps.append((b['updated_at'] - a['updated_at']).total_seconds())
            except Exception:
                pass
    feats += [mean(gaps) if gaps else 0.0, pstdev(gaps) if len(gaps) > 1 else 0.0]
    return feats


def build_dataset(rows: Sequence[dict], min_history: int = MAX_HISTORY):
    X, y = [], []
    for i in range(min_history, len(rows)):
        X.append(make_features(rows[:i]))
        y.append(_result_num(rows[i]))
    return X, y
