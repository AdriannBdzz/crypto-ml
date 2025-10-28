"""
Capa de datos: conectores y funciones de ingesta.
Expone:
- fetch_ohlcv: descarga OHLCV vía CCXT
"""

from .fetch_ccxt import fetch_ohlcv

__all__ = ["fetch_ohlcv"]
