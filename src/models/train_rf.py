import argparse
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, roc_auc_score, brier_score_loss
from pathlib import Path
import json

from src.utils.io import load_yaml, ensure_dir
from src.utils.timecv import purged_walk_forward_splits
from src.data.fetch_ccxt import fetch_ohlcv
from src.features.pipeline import make_features, make_labels
from src.backtest.simple_backtest import backtest_dir_signal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--horizon", type=int, default=None)

    # parámetros de validación (puedes cambiarlos por CLI)
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=200)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--embargo", type=int, default=None)  # por defecto = horizon
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--slip", type=float, default=0.0002)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    symbol = args.symbol or cfg["data"]["symbol"]
    exchange_id = args.exchange or cfg["data"]["exchange_id"]
    timeframe = args.timeframe or cfg["data"]["timeframe"]
    horizon = args.horizon or cfg["label"]["horizon"]
    embargo = args.embargo if args.embargo is not None else horizon

    print(f"[data] {symbol} @ {exchange_id} {timeframe}  limit={cfg['data']['limit']}")
    raw = fetch_ohlcv(
    symbol=symbol,
    exchange_id=exchange_id,
    timeframe=timeframe,
    limit=cfg["data"]["limit"],
    max_candles=cfg["data"].get("max_candles", None)  # <-- usa max_candles del YAML si existe
    )
    print(f"[data] fetched rows={len(raw)}")

    f = make_features(raw, **cfg["features"])
    X, y = make_labels(f, horizon=horizon)
    close = f.loc[X.index, "close"]  # precios alineados con X

    rf_cfg = cfg["model"]["rf"]
    model = RandomForestClassifier(**rf_cfg)

    print(f"[wfv] purged_walk_forward: train={args.train_size}, test={args.test_size}, step={args.step}, purge={horizon}, embargo={embargo}")
    folds = list(purged_walk_forward_splits(len(X), args.train_size, args.test_size, args.step,
                                            purge=horizon, embargo=embargo))
    if not folds:
        raise ValueError("No hay suficientes datos para los parámetros de WFV. Reduce train/test/step o aumenta limit.")

    fold_metrics = []
    for k, (tr, te) in enumerate(folds, start=1):
        m = RandomForestClassifier(**rf_cfg)
        m.fit(X.iloc[tr], y.iloc[tr])
        proba = m.predict_proba(X.iloc[te])[:, 1]
        pred = (proba > args.threshold).astype(int)
        yt = y.iloc[te]

        # métricas pred
        acc = accuracy_score(yt, pred)
        f1 = f1_score(yt, pred)
        mcc = matthews_corrcoef(yt, pred)
        try:
            auc = roc_auc_score(yt, proba)
        except Exception:
            auc = float("nan")
        brier = brier_score_loss(yt, proba)

        # backtest fold
        proba_s = proba if hasattr(proba, "index") else None
        # index de test
        proba_up = (yt * 0.0) + proba  # crea serie con mismo index que y_test
        proba_up.index = yt.index
        equity, rets, cost = backtest_dir_signal(close=close.loc[proba_up.index], proba_up=proba_up,
                                                 th=args.threshold, fee=args.fee, slip=args.slip)
        # sharpe-like
        ann = 24 if timeframe.endswith("h") else 365
        sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(ann) if rets.std() > 0 else float("nan")

        fm = {
            "fold": k,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "metrics": {"acc": acc, "f1": f1, "mcc": mcc, "auc": auc, "brier": brier},
            "backtest": {"final_equity": float(equity.iloc[-1]), "sharpe_like": float(sharpe)}
        }
        fold_metrics.append(fm)
        print(f"[fold {k:02d}] acc={acc:.3f} mcc={mcc:.3f} auc={auc:.3f} brier={brier:.3f} | eq={equity.iloc[-1]:.3f} sharpe={sharpe:.2f}")

    # resumen
    def avg(key): 
        vals = [fm["metrics"][key] for fm in fold_metrics if not np.isnan(fm["metrics"][key])]
        return float(np.mean(vals)) if vals else float("nan")
    def avg_bt(key):
        vals = [fm["backtest"][key] for fm in fold_metrics if not np.isnan(fm["backtest"][key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "symbol": symbol, "exchange": exchange_id, "timeframe": timeframe, "horizon": horizon,
        "wfv": {"train_size": args.train_size, "test_size": args.test_size, "step": args.step, "purge": horizon, "embargo": embargo},
        "cv_metrics_mean": {"acc": avg("acc"), "f1": avg("f1"), "mcc": avg("mcc"), "auc": avg("auc"), "brier": avg("brier")},
        "cv_backtest_mean": {"final_equity": avg_bt("final_equity"), "sharpe_like": avg_bt("sharpe_like")},
        "folds": fold_metrics
    }

    # entrena modelo final en TODO el histórico (opcional: podría ser en última ventana)
    model.fit(X, y)
    ensure_dir("artifacts")
    art_path = f"artifacts/rf_{exchange_id}_{symbol.replace('/','-')}_{timeframe}_{horizon}h.joblib"
    dump({"model": model, "features": list(X.columns), "meta": {
        "symbol": symbol, "exchange": exchange_id, "timeframe": timeframe, "horizon": horizon
    }}, art_path)
    print(f"[save] {art_path}")

    rep_path = Path("artifacts") / f"wfv_report_{exchange_id}_{symbol.replace('/','-')}_{timeframe}_{horizon}h.json"
    with open(rep_path, "w") as fjson:
        json.dump(summary, fjson, indent=2)
    print(f"[report] {rep_path}")

if __name__ == "__main__":
    main()
