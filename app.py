from __future__ import annotations

import threading

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from brain import db
from brain.worker import worker_loop
from brain.trainer import load_active_model, predict

app = FastAPI(title='Denius Train AI', version='1.1.0')

_worker_thread: threading.Thread | None = None


@app.on_event('startup')
def startup():
    global _worker_thread

    # The web service owns the long-running collector/training loop so the
    # deployment only needs one Render Web Service. Database access is still
    # persistent through DATABASE_URL.
    db.init_db()

    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(
            target=worker_loop,
            name='denius-train-worker',
            daemon=True,
        )
        _worker_thread.start()


@app.get('/health')
def health():
    return {
        'ok': True,
        'rounds': db.count_rounds(),
        'worker_alive': bool(
            _worker_thread and _worker_thread.is_alive()
        ),
    }


@app.get('/api/status')
def status():
    active = load_active_model()
    latest = db.latest_round()
    stats = db.prediction_stats()
    rows = db.all_rounds()
    current_prediction = predict(rows, active) if active and rows else None
    return {
        'ok': True,
        'source': 'kwinstore sunwin tx history',
        'rounds_stored': len(rows),
        'latest_session': latest['session'] if latest else None,
        'latest_result': latest['result'] if latest else None,
        'worker_alive': bool(
            _worker_thread and _worker_thread.is_alive()
        ),
        'model': ({
            'version': active['version'],
            'name': active['model_name'],
            'trained_rows': active['trained_rows'],
            'trained_at': active['trained_at'],
            'metrics': active['metrics']
        } if active else None),
        'live_prediction': current_prediction,
        'prediction_stats': stats,
    }


@app.get('/api/history')
def history(limit: int = 100):
    limit = max(1, min(int(limit), 1000))
    rows = db.all_rounds()
    return {'data': list(reversed(rows[-limit:]))}


@app.get('/api/predictions')
def predictions(limit: int = 100):
    limit = max(1, min(int(limit), 1000))
    return {'data': db.recent_predictions(limit)}


@app.get('/', response_class=HTMLResponse)
def dashboard():
    return HTML_PAGE


HTML_PAGE = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DENIUS TRAIN AI</title>
<style>
:root{font-family:Inter,system-ui,Arial;background:#080b12;color:#eef2ff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#141d35,#070910 55%);min-height:100vh}.wrap{max-width:1200px;margin:auto;padding:24px}.hero{padding:24px 0}.hero h1{margin:0 0 8px;font-size:34px}.muted{color:#9aa6bd}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:rgba(18,24,39,.82);border:1px solid #25304a;border-radius:18px;padding:18px;box-shadow:0 12px 40px rgba(0,0,0,.25)}.value{font-size:24px;font-weight:800;margin-top:8px}.wide{grid-column:span 2}.full{grid-column:1/-1}.pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#18233b}.ok{color:#6ef5ae}.warn{color:#ffd166}.err{color:#ff7b93}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:9px;border-bottom:1px solid #222b40;text-align:left}th{color:#9aa6bd}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:pre-wrap}.bar{height:10px;background:#202a40;border-radius:99px;overflow:hidden}.bar>i{display:block;height:100%;background:linear-gradient(90deg,#56d8ff,#b78cff);width:50%}@media(max-width:800px){.grid{grid-template-columns:1fr 1fr}.wide{grid-column:span 2}}@media(max-width:520px){.grid{grid-template-columns:1fr}.wide,.full{grid-column:auto}}
</style></head><body><div class="wrap"><div class="hero"><h1>🧠 DENIUS TRAIN AI</h1><div class="muted">Lưu lịch sử bền vững · huấn luyện theo thời gian · walk-forward validation · champion/challenger · collector chạy nền</div></div>
<div class="grid"><div class="card"><div class="muted">Lịch sử đã lưu</div><div id="rounds" class="value">—</div></div><div class="card"><div class="muted">Phiên mới nhất</div><div id="session" class="value">—</div></div><div class="card"><div class="muted">Model hiện tại</div><div id="model" class="value" style="font-size:16px">—</div></div><div class="card"><div class="muted">Accuracy dự đoán đã giải</div><div id="acc" class="value">—</div></div><div class="card wide"><div class="muted">Dự đoán phiên kế</div><div id="pred" class="value">Chưa có model</div><div id="prob" class="muted" style="margin-top:10px"></div><div class="bar" style="margin-top:12px"><i id="pbar"></i></div></div><div class="card wide"><div class="muted">Trạng thái huấn luyện</div><div id="train" class="mono" style="margin-top:10px">Đang tải…</div></div><div class="card full"><div class="muted" style="margin-bottom:10px">Dự đoán gần đây</div><table><thead><tr><th>Nguồn</th><th>Mục tiêu</th><th>Dự đoán</th><th>P(Tài)</th><th>Kết quả</th><th>Đúng?</th></tr></thead><tbody id="tbody"></tbody></table></div></div></div>
<script>
const $=id=>document.getElementById(id); const esc=x=>String(x??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
async function load(){try{const s=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json()); $('rounds').textContent=s.rounds_stored.toLocaleString(); $('session').textContent=s.latest_session??'—'; const m=s.model; $('model').textContent=m?m.name+' · '+m.version.split('-').slice(-2,-1)[0]:'Đang train'; const a=s.prediction_stats.accuracy; $('acc').textContent=a==null?'—':(a*100).toFixed(2)+'%'; const p=s.live_prediction; if(p){$('pred').textContent=p.predicted_result+' · '+(p.confidence*100).toFixed(2)+'%';$('prob').textContent='P(Tài) '+(p.p_tai*100).toFixed(2)+'% · P(Xỉu) '+(p.p_xiu*100).toFixed(2)+'%';$('pbar').style.width=(p.p_tai*100)+'%'} else {$('pred').textContent='Chưa đủ dữ liệu / model'} $('train').textContent=m?JSON.stringify({name:m.name,trained_rows:m.trained_rows,trained_at:m.trained_at,worker_alive:s.worker_alive,validation:m.metrics.evaluations[m.metrics.selected]},null,2):'Chưa có model'; const pr=await fetch('/api/predictions?limit=30',{cache:'no-store'}).then(r=>r.json()); $('tbody').innerHTML=pr.data.map(x=>`<tr><td>${esc(x.source_session)}</td><td>${esc(x.target_session)}</td><td>${esc(x.predicted_result)}</td><td>${(x.p_tai*100).toFixed(2)}%</td><td>${esc(x.actual_result??'chờ')}</td><td>${x.is_correct==null?'—':(x.is_correct?'✅':'❌')}</td></tr>`).join('');}catch(e){$('train').textContent='Lỗi dashboard: '+e} } load(); setInterval(load,5000);
</script></body></html>'''
