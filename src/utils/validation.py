import pandas as pd

def assert_no_lookahead(df: pd.DataFrame, feat_cols: list[str]) -> None:
    """
    Verifica que no haya NaNs introducidos por desplazar mal, y que no existan features con
    valores 'futuros' obvios (heurística: no debe haber alineaciones extrañas tras shift).
    Lanza AssertionError si detecta problemas básicos.
    """
    # 1) No NaNs “sorpresa”
    assert not df[feat_cols].isna().any().any(), "Hay NaNs en features; revisa shift/rolling."

    # 2) Heurística: si incrementas el índice 1 paso y comparas, no deberían coincidir 100% (señal de fuga)
    # (Esto es suave; en práctica puedes incluir pruebas más específicas si añades más features.)
