"""
Modelos y entrenamiento.
Expone atajos para lanzar entrenamientos programáticamente:

from src.models import train_random_forest
model_path = train_random_forest(symbol="BTC/USDT")

(Para baselines o deep learning, usa los módulos correspondientes.)
"""

from __future__ import annotations
from pathlib import Path
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import matthews_corrcoef
import numpy as np

from src.utils import load_yaml, ensure_dir
from src.data import fetch_ohlcv
from src.features import make_features, make_labels

__all__ = ["train_random_forest"]

def train_random_forest(config_path: str = "configs/settings.yaml",
                        symbol: str | None = None,
                        exchange_id: str | None = None,
                        timeframe: str | None = None,
                        horizon: int | None = None) -> str:
    cfg = load_yaml(config_path)
    symbol = symbol or cfg["data"]["symbol"]
    exchange_id = exchange_id or cfg["data"]["exchange_id"]
    timeframe = timeframe or cfg["data"]["timeframe"]
    horizon = horizon or cfg["label"]["horizon"]

    raw = fetch_ohlcv(symbol=symbol, exchange_id=exchange_id, timeframe=timeframe, limit=cfg["data"]["limit"])
    feat = make_features(raw, **cfg["features"])
    X, y = make_labels(feat, horizon=horizon)

    model = RandomForestClassifier(**cfg["model"]["rf"])
    tscv = TimeSeriesSplit(n_splits=5)
    accs, mccs = [], []
    for tr, va in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict(X.iloc[va])
        accs.append((p == y.iloc[va]).mean())
        mccs.append(matthews_corrcoef(y.iloc[va], p))
    print(f"[cv] Acc={np.mean(accs):.4f} MCC={np.mean(mccs):.4f}")

    model.fit(X, y)
    ensure_dir("artifacts")
    out = f"artifacts/rf_{exchange_id}_{symbol.replace('/','-')}_{timeframe}_{horizon}h.joblib"
    dump({"model": model, "features": list(X.columns), "meta": {
        "symbol": symbol, "exchange": exchange_id, "timeframe": timeframe, "horizon": horizon
    }}, out)
    return out
