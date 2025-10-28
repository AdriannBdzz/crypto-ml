from typing import Iterator, Tuple
import numpy as np

def walk_forward_splits(n: int, train_size: int, test_size: int, step: int) -> Iterator[Tuple[range, range]]:
    """
    Split simple (sin purge/embargo). Mantengo por compatibilidad.
    """
    start = 0
    while start + train_size + test_size <= n:
        yield range(start, start + train_size), range(start + train_size, start + train_size + test_size)
        start += step

def purged_walk_forward_splits(n: int, train_size: int, test_size: int, step: int,
                               purge: int = 0, embargo: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward con purge y embargo:
      - purge: elimina las últimas 'purge' observaciones del bloque de entrenamiento
               para que las etiquetas que miran t+h no alcancen el bloque de test.
      - embargo: separa train y test con un hueco adicional.
    Devuelve índices (np.ndarray) para .iloc.
    """
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_size  # exclusivo
        test_start = train_end + embargo
        test_end = test_start + test_size

        # Condición de parada
        if test_end > n:
            break

        # Aplicar purge al train
        tr_end_purged = max(train_start, train_end - purge)
        train_idx = np.arange(train_start, tr_end_purged, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)

        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx

        start += step
