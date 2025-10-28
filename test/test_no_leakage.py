import pandas as pd
import numpy as np
from src.features.pipeline import make_features, make_labels

def test_no_leakage_shift_open_next():
    # Serie artificial monotónica: si hubiera fuga, el modelo “adivinaría” perfecto.
    idx = pd.date_range("2024-01-01", periods=200, freq="H")
    close = pd.Series(np.arange(200), index=idx, name="close")
    df = pd.DataFrame({
        "open": close.values,
        "high": close.values + 1,
        "low": close.values - 1,
        "close": close.values,
        "volume": 1000
    }, index=idx)
    f = make_features(df, decision_at="open_next")  # aseguramos shift
    X, y = make_labels(f, horizon=6)
    # si hubiera fuga, X e y estarían casi perfectamente correlacionados en una serie monotónica
    corr = pd.Series(X["ret_1"]).corr(pd.Series(y, index=X.index))
    assert abs(corr) < 0.5, "Posible fuga detectada en features vs labels."
