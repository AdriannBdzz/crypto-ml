import argparse, numpy as np, pandas as pd
from pathlib import Path
from joblib import load
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                             matthews_corrcoef, brier_score_loss)
from scipy.stats import ks_2samp

from src.data.fetch_ccxt import fetch_ohlcv
from src.features.pipeline import make_features, make_labels
from src.backtest.simple_backtest import backtest_dir_signal
from src.utils.io import load_yaml

def _safe_auc(y_true, p):
    try:
        return roc_auc_score(y_true, p)
    except Exception:
        return np.nan

def evaluate_fixed_model(artifact_path: str, symbol: str, exchange_id: str,
                         timeframe: str, horizon: int,
                         limit: int = 2000, th: float = 0.55,
                         fee: float = 0.0004, slip: float = 0.0002):
    # cargar artefacto y config
    art = load(artifact_path)
    model, feat_names, meta = art["model"], art["features"], art["meta"]

    # datos y features
    df = fetch_ohlcv(symbol=symbol, exchange_id=exchange_id, timeframe=timeframe, limit=limit)
    f = make_features(df)
    f = f.dropna()
    # etiquetas para evaluación ex-post (necesitan futuro)
    X_all, y_all = make_labels(f, horizon=horizon)

    # alineamos a features disponibles en tiempo real
    X = X_all[feat_names].copy()
    y = y_all.copy()

    # predicciones “pseudo-live” (modelo fijo)
    proba_up = pd.Series(model.predict_proba(X)[:,1], index=X.index, name="proba_up")
    pred = (proba_up > 0.5).astype(int)

    # métricas predictivas
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    mcc = matthews_corrcoef(y, pred)
    auc = _safe_auc(y, proba_up)
    brier = brier_score_loss(y, proba_up)

    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()

    # backtest económico sencillo
    equity, ret, cost = backtest_dir_signal(close=f.loc[proba_up.index, "close"], proba_up=proba_up, th=th, fee=fee, slip=slip)
    rets = ret.replace([np.inf,-np.inf], 0).fillna(0)
    if rets.std() > 0:
        sharpe = rets.mean() / rets.std() * np.sqrt(24 if timeframe.endswith("h") else 365)
    else:
        sharpe = np.nan
    # max drawdown
    roll_max = equity.cummax()
    drawdown = (equity / roll_max - 1.0)
    mdd = drawdown.min()

    # drift simple (KS) comparando primeras 500 obs vs. últimas 500
    drift = {}
    wnd = min(500, len(X))
    if wnd > 50:
        A, B = X.iloc[:wnd], X.iloc[-wnd:]
        for c in X.columns:
            try:
                stat, pval = ks_2samp(A[c].dropna(), B[c].dropna())
                drift[c] = {"ks_stat": float(stat), "p_value": float(pval)}
            except Exception:
                drift[c] = {"ks_stat": np.nan, "p_value": np.nan}

    report = {
        "meta": {
            "symbol": symbol, "exchange": exchange_id, "timeframe": timeframe,
            "horizon": horizon, "n_samples": int(len(X))
        },
        "metrics": {
            "accuracy": float(acc), "f1": float(f1), "mcc": float(mcc),
            "auc": float(auc) if auc==auc else None,
            "brier": float(brier),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        },
        "backtest": {
            "threshold": th, "fee": fee, "slippage": slip,
            "final_equity": float(equity.iloc[-1]),
            "sharpe_like": float(sharpe) if sharpe==sharpe else None,
            "max_drawdown": float(mdd)
        },
        "data_health": {
            "nan_in_X": int(X.isna().sum().sum()),
            "nan_in_y": int(pd.isna(y).sum()),
            "drift_ks": drift
        }
    }
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=None, help="ruta al .joblib del modelo")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--slip", type=float, default=0.0002)
    args = ap.parse_args()

    if args.artifact is None:
        # construir ruta por defecto
        art = f"artifacts/rf_{args.exchange}_{args.symbol.replace('/','-')}_{args.timeframe}_{args.horizon}h.joblib"
    else:
        art = args.artifact

    rep = evaluate_fixed_model(artifact_path=art,
                               symbol=args.symbol, exchange_id=args.exchange,
                               timeframe=args.timeframe, horizon=args.horizon,
                               limit=args.limit, th=args.threshold,
                               fee=args.fee, slip=args.slip)
    import json; print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
