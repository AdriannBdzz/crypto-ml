"""
Crypto-ML: Paquete principal de la aplicación.

Expone:
- __version__: versión del paquete
- get_logger(name): helper para obtener loggers con formato consistente
"""

from __future__ import annotations
import logging
from typing import Optional

__all__ = ["__version__", "get_logger"]

__version__ = "0.1.0"

_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def _ensure_basic_logging(level: int = logging.INFO) -> None:
    # Evita duplicar handlers si el root ya está configurado (p.ej., en uvicorn)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    _ensure_basic_logging()
    return logging.getLogger(name or "crypto_ml")
