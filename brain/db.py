from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable
import json
import psycopg
from psycopg.rows import dict_row

from .config import DATABASE_URL

SCHEMA = '''
CREATE TABLE IF NOT EXISTS rounds (
    session BIGINT PRIMARY KEY,
    d1 SMALLINT NOT NULL,
    d2 SMALLINT NOT NULL,
    d3 SMALLINT NOT NULL,
    total SMALLINT NOT NULL,
    result TEXT NOT NULL CHECK (result IN ('Tài','Xỉu')),
    updated_at TIMESTAMPTZ NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS rounds_updated_idx ON rounds(updated_at DESC);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    source_session BIGINT NOT NULL,
    target_session BIGINT NOT NULL,
    predicted_result TEXT NOT NULL CHECK (predicted_result IN ('Tài','Xỉu')),
    p_tai DOUBLE PRECISION NOT NULL,
    p_xiu DOUBLE PRECISION NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actual_result TEXT NULL,
    is_correct BOOLEAN NULL,
    brier DOUBLE PRECISION NULL,
    UNIQUE(source_session, model_version)
);
CREATE INDEX IF NOT EXISTS predictions_pending_idx ON predictions(actual_result, source_session DESC);

CREATE TABLE IF NOT EXISTS model_state (
    id SMALLINT PRIMARY KEY CHECK (id = 1),
    model_version TEXT NOT NULL,
    artifact BYTEA NOT NULL,
    metrics JSONB NOT NULL,
    trained_rows INTEGER NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS brain_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
'''

@contextmanager
def conn():
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as c:
        yield c


def init_db() -> None:
    with conn() as c:
        c.execute(SCHEMA)
        c.commit()


def upsert_rounds(rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    inserted = 0
    with conn() as c:
        for r in rows:
            cur = c.execute(
                '''INSERT INTO rounds(session,d1,d2,d3,total,result,updated_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (session) DO UPDATE SET
                     d1=EXCLUDED.d1,d2=EXCLUDED.d2,d3=EXCLUDED.d3,
                     total=EXCLUDED.total,result=EXCLUDED.result,updated_at=EXCLUDED.updated_at
                   WHERE rounds.d1<>EXCLUDED.d1 OR rounds.d2<>EXCLUDED.d2 OR rounds.d3<>EXCLUDED.d3
                      OR rounds.total<>EXCLUDED.total OR rounds.result<>EXCLUDED.result
                      OR rounds.updated_at IS DISTINCT FROM EXCLUDED.updated_at''',
                (r['session'], r['d1'], r['d2'], r['d3'], r['total'], r['result'], r.get('updated_at')),
            )
            inserted += cur.rowcount
        c.commit()
    return inserted


def all_rounds() -> list[dict[str, Any]]:
    with conn() as c:
        return c.execute('''SELECT session,d1,d2,d3,total,result,updated_at FROM rounds ORDER BY session ASC''').fetchall()


def latest_round() -> dict[str, Any] | None:
    with conn() as c:
        return c.execute('''SELECT session,d1,d2,d3,total,result,updated_at FROM rounds ORDER BY session DESC LIMIT 1''').fetchone()


def count_rounds() -> int:
    with conn() as c:
        return c.execute('SELECT COUNT(*) AS n FROM rounds').fetchone()['n']


def save_model(model_version: str, artifact: bytes, metrics: dict, trained_rows: int) -> None:
    with conn() as c:
        c.execute('''INSERT INTO model_state(id,model_version,artifact,metrics,trained_rows)
                     VALUES (1,%s,%s,%s,%s)
                     ON CONFLICT(id) DO UPDATE SET model_version=EXCLUDED.model_version,
                       artifact=EXCLUDED.artifact,metrics=EXCLUDED.metrics,
                       trained_rows=EXCLUDED.trained_rows,trained_at=NOW()''',
                  (model_version, artifact, json.dumps(metrics), trained_rows))
        c.commit()


def load_model():
    with conn() as c:
        return c.execute('SELECT model_version,artifact,metrics,trained_rows,trained_at FROM model_state WHERE id=1').fetchone()


def create_prediction(source_session: int, target_session: int, predicted_result: str,
                      p_tai: float, model_version: str) -> None:
    p_xiu = 1.0 - p_tai
    with conn() as c:
        c.execute('''INSERT INTO predictions(source_session,target_session,predicted_result,p_tai,p_xiu,model_version)
                     VALUES (%s,%s,%s,%s,%s,%s)
                     ON CONFLICT (source_session,model_version) DO NOTHING''',
                  (source_session,target_session,predicted_result,p_tai,p_xiu,model_version))
        c.commit()


def resolve_predictions(actual_session: int, actual_result: str) -> int:
    with conn() as c:
        rows = c.execute('''SELECT id,p_tai,predicted_result FROM predictions
                            WHERE target_session=%s AND actual_result IS NULL''', (actual_session,)).fetchall()
        for row in rows:
            is_correct = row['predicted_result'] == actual_result
            p = row['p_tai'] if actual_result == 'Tài' else 1-row['p_tai']
            brier = (row['p_tai'] - (1.0 if actual_result == 'Tài' else 0.0)) ** 2
            c.execute('''UPDATE predictions SET actual_result=%s,is_correct=%s,brier=%s WHERE id=%s''',
                      (actual_result, is_correct, brier, row['id']))
        c.commit()
        return len(rows)


def recent_predictions(limit: int = 50) -> list[dict[str, Any]]:
    with conn() as c:
        return c.execute('''SELECT source_session,target_session,predicted_result,p_tai,model_version,
                                   created_at,actual_result,is_correct,brier
                            FROM predictions ORDER BY id DESC LIMIT %s''', (limit,)).fetchall()


def prediction_stats() -> dict[str, Any]:
    with conn() as c:
        row = c.execute('''SELECT COUNT(*) FILTER (WHERE actual_result IS NOT NULL) AS resolved,
                                  COUNT(*) FILTER (WHERE actual_result IS NOT NULL AND is_correct) AS correct,
                                  AVG(brier) FILTER (WHERE actual_result IS NOT NULL) AS brier
                           FROM predictions''').fetchone()
        resolved = int(row['resolved'] or 0)
        correct = int(row['correct'] or 0)
        return {'resolved': resolved, 'correct': correct, 'accuracy': (correct/resolved if resolved else None), 'brier': row['brier']}


def log_event(event_type: str, message: str) -> None:
    with conn() as c:
        c.execute('INSERT INTO brain_events(event_type,message) VALUES (%s,%s)', (event_type, message))
        c.commit()
