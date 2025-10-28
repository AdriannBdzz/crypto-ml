import pandas as pd

def backtest_dir_signal(close: pd.Series, proba_up: pd.Series, th=0.55, fee=0.0004, slip=0.0002):
    sig = (proba_up > th).astype(int).diff().fillna(0)
    pos = (proba_up > th).astype(int)
    ret = close.pct_change().fillna(0) * pos.shift(1).fillna(0)
    cost = sig.abs() * (fee + slip)
    equity = (1 + ret - cost).cumprod()
    return equity, ret, cost
