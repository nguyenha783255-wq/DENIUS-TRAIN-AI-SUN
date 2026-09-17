from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from .config import MIN_TRAIN_ROWS, RANDOM_STATE, MODEL_VERSION
from .features import build_dataset, make_features
from . import db


def candidate_models():
    return {
        'logreg': Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(C=0.6, max_iter=1500, class_weight='balanced', random_state=RANDOM_STATE))
        ]),
        'extratrees': ExtraTreesClassifier(
            n_estimators=350, max_features='sqrt', min_samples_leaf=4,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        ),
        'histgb': HistGradientBoostingClassifier(
            learning_rate=0.045, max_iter=250, max_leaf_nodes=15,
            l2_regularization=1.5, random_state=RANDOM_STATE
        )
    }


def evaluate_walk_forward(model, X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    if n < 400:
        return None
    test_size = max(50, n // 8)
    tscv = TimeSeriesSplit(n_splits=4, test_size=test_size)
    folds = []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        folds.append({
            'accuracy': float(accuracy_score(y[test_idx], pred)),
            'logloss': float(log_loss(y[test_idx], prob, labels=[0,1])),
            'brier': float(brier_score_loss(y[test_idx], prob)),
            'n': int(len(test_idx)),
        })
    return {
        'accuracy': float(np.mean([f['accuracy'] for f in folds])),
        'logloss': float(np.mean([f['logloss'] for f in folds])),
        'brier': float(np.mean([f['brier'] for f in folds])),
        'folds': folds,
    }


def train_champion(rows: list[dict[str, Any]]):
    if len(rows) < MIN_TRAIN_ROWS:
        raise RuntimeError(f'Need at least {MIN_TRAIN_ROWS} rounds, have {len(rows)}')
    X, y = build_dataset(rows)
    results = {}
    champions = candidate_models()
    for name, model in champions.items():
        results[name] = evaluate_walk_forward(model, X, y)

    # Primary criterion: lowest temporal log-loss; accuracy/brier are retained for transparency.
    name = min(results, key=lambda k: (results[k]['logloss'], results[k]['brier'], -results[k]['accuracy']))
    model = champions[name]
    model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))

    metrics = {
        'selected': name,
        'rows': len(rows),
        'samples': len(y),
        'evaluations': results,
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'selection_rule': 'temporal walk-forward: lowest mean log-loss, then brier, then higher accuracy',
    }
    bio = io.BytesIO()
    joblib.dump({'model': model, 'model_name': name, 'feature_version': 'v1'}, bio, compress=3)
    version = f'{MODEL_VERSION}-{name}-{datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")}'
    return version, bio.getvalue(), metrics


def train_and_save(rows: list[dict[str, Any]]):
    version, artifact, metrics = train_champion(rows)
    db.save_model(version, artifact, metrics, len(rows))
    db.log_event('TRAIN', f'Champion={metrics["selected"]}; rows={len(rows)}; accuracy={metrics["evaluations"][metrics["selected"]]["accuracy"]:.4f}; logloss={metrics["evaluations"][metrics["selected"]]["logloss"]:.4f}')
    return version, metrics


def load_active_model():
    state = db.load_model()
    if not state:
        return None
    obj = joblib.load(io.BytesIO(state['artifact']))
    return {
        'version': state['model_version'],
        'metrics': state['metrics'],
        'trained_rows': state['trained_rows'],
        'model': obj['model'],
        'model_name': obj['model_name'],
        'feature_version': obj['feature_version'],
        'trained_at': state['trained_at'],
    }


def predict(rows: list[dict[str, Any]], active: dict[str, Any] | None):
    if active is None or len(rows) < 100:
        return None
    X = np.asarray([make_features(rows)], dtype=float)
    p = float(active['model'].predict_proba(X)[0,1])
    pred = 'Tài' if p >= 0.5 else 'Xỉu'
    return {'predicted_result': pred, 'p_tai': p, 'p_xiu': 1-p,
            'confidence': max(p, 1-p), 'model_version': active['version']}
