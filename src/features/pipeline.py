import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator

def make_features(
    df: pd.DataFrame,
    rsi_window=14, ema_fast=21, ema_slow=55,
    macd_fast=12, macd_slow=26, macd_signal=9,
    rvol_window=24, rv_window=24,
    decision_at: str = "close",           # "close" o "open_next"
    shift_features_by_1: bool = False     # si True, desplaza 1 barra todos los features
) -> pd.DataFrame:
    """
    Genera features usando solo info ≤ t (rolling/indicators). Por defecto, se asume que decides al cierre de t.
    Si decision_at='open_next' o shift_features_by_1=True, desplazamos los features 1 barra para evitar
    usar la vela t al decidir en la apertura de t+1.
    """
    f = df.copy()

    # 1) Retornos/volatilidad realizados hasta t (no usan futuro)
    f["ret_1"] = np.log(f["close"]).diff(1)

    # Volatilidad realizada a 24h (≈ 1 día)
    f["rv"] = f["ret_1"].rolling(rv_window, min_periods=rv_window).std() * np.sqrt(rv_window)

    # Volatilidad realizada a 7d (24*7=168 velas si timeframe=1h)
    f["rv_7d"] = f["ret_1"].rolling(24*7, min_periods=24*7).std() * np.sqrt(24*7)

    # 2) Indicadores técnicos (usan <= t internamente)
    rsi = RSIIndicator(f["close"], window=rsi_window)
    f["rsi"] = rsi.rsi()

    macd = MACD(f["close"], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    f["macd"] = macd.macd()
    f["macd_sig"] = macd.macd_signal()

    ema_f = EMAIndicator(f["close"], window=ema_fast).ema_indicator()
    ema_s = EMAIndicator(f["close"], window=ema_slow).ema_indicator()
    f["ema_fast"] = ema_f
    f["ema_slow"] = ema_s
    f["ema_spread"] = (f["ema_fast"] - f["ema_slow"]) / f["close"]

    # 3) Volumen relativo hasta t
    f["rvol"] = f["volume"] / f["volume"].rolling(rvol_window, min_periods=rvol_window).mean()

    # 4) Si la decisión es en la apertura de t+1, desplazamos los features una barra hacia adelante
    if decision_at == "open_next" or shift_features_by_1:
        cols = ["ret_1","rv","rsi","macd","macd_sig","ema_fast","ema_slow","ema_spread","rvol"]
        f[cols] = f[cols].shift(1)

    # 5) Dropna de forma consistente (evita que ventanas incompletas filtren distinto por split)
    f = f.dropna()

    return f

def make_labels(feat: pd.DataFrame, horizon=6):
    """
    Clasificación direccional a horizonte h:
    y=1 si log(close_{t+h}/close_t) > 0
    """
    y_reg = np.log(feat["close"].shift(-horizon)) - np.log(feat["close"])
    y_cls = (y_reg > 0).astype(int)

    X = feat.drop(columns=["open","high","low","close","volume"])
    # Alinear para que X_t use label en t+h
    X = X.iloc[:-horizon]
    y_cls = y_cls.iloc[:-horizon]
    return X, y_cls
