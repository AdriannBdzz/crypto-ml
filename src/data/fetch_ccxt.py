import time
from typing import Literal, Optional

import ccxt
import pandas as pd


def _tf_ms(exchange, timeframe: str) -> int:
    # ccxt ya expone parse_timeframe (en segundos)
    return exchange.parse_timeframe(timeframe) * 1000


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    exchange_id: str = "binance",
    timeframe: Literal["1m", "5m", "15m", "1h", "4h", "1d"] = "1h",
    limit: int = 1000,              # tamaño por llamada
    since: Optional[int] = None,    # timestamp en ms
    max_candles: Optional[int] = None,  # total deseado
    sleep_sec: float = 0.2,
) -> pd.DataFrame:
    """
    Descarga OHLCV con paginación hacia ADELANTE usando 'since'.
    - limit: tamaño del lote por llamada (Binance suele permitir 1000).
    - max_candles: total deseado (p. ej., 5000). Si None, trae solo un lote.
    - since: si no se da, se calcula para cubrir aprox. max_candles hacia atrás.
    """
    ex_cls = getattr(ccxt, exchange_id)
    ex = ex_cls({"enableRateLimit": True})

    # Si no quieres paginar, respeta el comportamiento clásico:
    if not max_candles or max_candles <= limit:
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
        if not data:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("UTC")
        df = df.set_index("timestamp").drop(columns=["ts"]).sort_index()
        return df

    tfms = _tf_ms(ex, timeframe)
    now_ms = ex.milliseconds()

    # Si no nos pasan 'since', estimamos para cubrir max_candles hacia atrás
    if since is None:
        approx_span = max_candles * tfms
        # resta un margen extra de 5 velas por seguridad
        since = now_ms - approx_span - 5 * tfms

    all_rows = []
    fetched = 0
    cur_since = since
    last_last_ts = None

    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cur_since, limit=limit)
        if not batch:
            break

        # Evita bucles si el exchange repite la última vela
        if last_last_ts is not None and batch[-1][0] <= last_last_ts:
            break
        last_last_ts = batch[-1][0]

        all_rows.extend(batch)
        fetched += len(batch)

        if max_candles and fetched >= max_candles:
            break

        # avanza el cursor al siguiente bloque
        cur_since = batch[-1][0] + tfms

        # si el exchange devolvió menos que el límite, probablemente no hay más datos
        if len(batch) < limit:
            break

        time.sleep(sleep_sec)  # respeta rate limits

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    # elimina duplicados y ordena
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert("UTC")
    df = df.set_index("timestamp").drop(columns=["ts"]).sort_index()

    # Si pedimos más de lo que hay, df puede ser < max_candles (esto es normal)
    return df
