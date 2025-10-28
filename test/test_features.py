import pandas as pd
from src.features.pipeline import make_features, make_labels

def test_make_features_and_labels():
    # datos sintéticos mínimos
    idx = pd.date_range("2024-01-01", periods=200, freq="H")
    df = pd.DataFrame({
        "open": 100 + range(200),
        "high": 101 + range(200),
        "low": 99 + range(200),
        "close": 100 + range(200),
        "volume": 1000
    }, index=idx)
    feat = make_features(df)
    X, y = make_labels(feat, horizon=6)
    assert not X.empty and len(X) == len(y)
