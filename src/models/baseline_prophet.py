import argparse
import pandas as pd
from prophet import Prophet
from src.utils.io import load_yaml
from src.data.fetch_ccxt import fetch_ohlcv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/settings.yaml")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--exchange", default=None)
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    symbol = args.symbol or cfg["data"]["symbol"]
    exchange_id = args.exchange or cfg["data"]["exchange_id"]
    timeframe = args.timeframe or cfg["data"]["timeframe"]

    df = fetch_ohlcv(symbol=symbol, exchange_id=exchange_id, timeframe=timeframe, limit=cfg["data"]["limit"])
    d = df.reset_index()[["timestamp","close"]].rename(columns={"timestamp":"ds","close":"y"})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.fit(d)
    future = m.make_future_dataframe(periods=args.horizon, freq="H" if timeframe.endswith("h") else "D")
    fcst = m.predict(future).tail(args.horizon)
    print(fcst[["ds","yhat","yhat_lower","yhat_upper"]].to_string(index=False))

if __name__ == "__main__":
    main()
