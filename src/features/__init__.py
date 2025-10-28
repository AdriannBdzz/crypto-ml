"""
Ingeniería de características.
Expone:
- make_features: crea features técnicos (RSI, MACD, EMAs, RVOL, etc.)
- make_labels: genera etiquetas (clasificación direccional por horizonte)
"""

from .pipeline import make_features, make_labels

FEATURES_PUBLIC = [
    "ret_1", "rv", "rsi", "macd", "macd_sig", "ema_fast", "ema_slow", "ema_spread", "rvol"
]

__all__ = ["make_features", "make_labels", "FEATURES_PUBLIC"]
