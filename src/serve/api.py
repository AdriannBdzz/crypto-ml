import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from joblib import load
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from src.data.fetch_ccxt import fetch_ohlcv
from src.features.pipeline import make_features
from src.utils.io import load_yaml

load_dotenv()

app = FastAPI(title="Crypto ML Predictor", version="0.1.0")

CFG = load_yaml("configs/settings.yaml")
DEFAULT_SYMBOL = CFG["data"]["symbol"]
DEFAULT_EXCHANGE = os.getenv("EXCHANGE_ID", CFG["data"]["exchange_id"])
DEFAULT_TIMEFRAME = CFG["data"]["timeframe"]
DEFAULT_HORIZON = CFG["label"]["horizon"]
THRESHOLD = CFG["serve"]["threshold_up"]

# Carga del modelo por defecto (entrenado previamente)
def _default_artifact_path():
    return f"artifacts/rf_{DEFAULT_EXCHANGE}_{DEFAULT_SYMBOL.replace('/','-')}_{DEFAULT_TIMEFRAME}_{DEFAULT_HORIZON}h.joblib"

ART = None
try:
    ART = load(_default_artifact_path())
    MODEL, FEATS, META = ART["model"], ART["features"], ART["meta"]
except Exception as e:
    MODEL, FEATS, META = None, None, None

class PredictionResponse(BaseModel):
    symbol: str
    timeframe: str
    horizon_hours: int
    prob_up: float | None
    prediction: int | None
    latest_price: float | None
    features_ts: str | None
    loaded_model: bool

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "features": FEATS,
        "version": app.version,
        "meta": META,
    }

@app.get("/predict", response_model=PredictionResponse)
def predict(symbol: str = Query(DEFAULT_SYMBOL),
           timeframe: str = Query(DEFAULT_TIMEFRAME),
           horizon_hours: int = Query(DEFAULT_HORIZON),
           exchange_id: str = Query(DEFAULT_EXCHANGE)):
    global MODEL, FEATS
    if MODEL is None:
        return PredictionResponse(symbol=symbol, timeframe=timeframe, horizon_hours=horizon_hours,
                                  prob_up=None, prediction=None, latest_price=None,
                                  features_ts=None, loaded_model=False)
    df = fetch_ohlcv(symbol=symbol, exchange_id=exchange_id, timeframe=timeframe, limit=400)
    f = make_features(df, **CFG["features"])
    f = f.dropna()
    X = f[FEATS].iloc[[-1]]
    proba_up = float(MODEL.predict_proba(X)[0,1])
    pred = int(proba_up > THRESHOLD)
    return PredictionResponse(
        symbol=symbol, timeframe=timeframe, horizon_hours=horizon_hours,
        prob_up=proba_up, prediction=pred,
        latest_price=float(df["close"].iloc[-1]),
        features_ts=str(f.index[-1].to_pydatetime()),
        loaded_model=True
    )
