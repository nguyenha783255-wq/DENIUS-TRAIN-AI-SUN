from __future__ import annotations

import time
from datetime import datetime
import threading
import httpx

from .config import SOURCE_URL, POLL_SECONDS, RETRAIN_EVERY_NEW, API_TIMEOUT
from . import db
from .trainer import train_and_save, load_active_model, predict


def normalize_item(x: dict) -> dict:
    from datetime import timezone
    ts = x.get('updatedAt')
    if ts:
        ts = ts.replace('Z', '+00:00')
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            dt = None
    else:
        dt = None
    return {
        'session': int(x['phiên']),
        'd1': int(x['d1']), 'd2': int(x['d2']), 'd3': int(x['d3']),
        'total': int(x['tổng']),
        'result': 'Tài' if str(x['kết quả']).strip().lower() == 'tài' else 'Xỉu',
        'updated_at': dt,
    }


def fetch_history() -> list[dict]:
    headers = {'User-Agent': 'DENIUS-TRAIN-AI/1.0', 'Accept': 'application/json'}
    with httpx.Client(timeout=API_TIMEOUT, headers=headers, follow_redirects=True) as client:
        r = client.get(SOURCE_URL, params={'t': int(time.time()*1000)})
        r.raise_for_status()
        payload = r.json()
    data = payload.get('data') if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        raise ValueError('API data is not a list')
    out = []
    for x in data:
        try:
            out.append(normalize_item(x))
        except Exception:
            continue
    # Source is newest-first; DB uses ascending session for training.
    out.sort(key=lambda x: x['session'])
    return out


def worker_loop(stop_event: threading.Event | None = None):
    db.init_db()
    active = load_active_model()
    last_seen = None
    since_retrain = 0
    first = True
    while stop_event is None or not stop_event.is_set():
        try:
            incoming = fetch_history()
            changed = db.upsert_rounds(incoming)
            latest = db.latest_round()
            if latest:
                if last_seen is None or latest['session'] != last_seen:
                    # First fetch seeds the whole table. Later fetches resolve the prior next-session prediction.
                    if not first:
                        db.resolve_predictions(latest['session'], latest['result'])
                    last_seen = latest['session']
            if changed:
                since_retrain += changed
                rows = db.all_rounds()
                if active is None or since_retrain >= RETRAIN_EVERY_NEW:
                    try:
                        version, metrics = train_and_save(rows)
                        active = load_active_model()
                        since_retrain = 0
                    except Exception as e:
                        db.log_event('TRAIN_ERROR', str(e))

                rows = db.all_rounds()
                if active and rows:
                    pr = predict(rows, active)
                    if pr:
                        source_session = rows[-1]['session']
                        target_session = source_session + 1
                        db.create_prediction(source_session, target_session, pr['predicted_result'], pr['p_tai'], pr['model_version'])
            first = False
        except Exception as e:
            db.log_event('POLL_ERROR', repr(e))
        if stop_event is not None:
            stop_event.wait(POLL_SECONDS)
        else:
            time.sleep(POLL_SECONDS)
