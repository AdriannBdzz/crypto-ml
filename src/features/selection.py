import pandas as pd

def drop_correlated_features(X: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Elimina columnas altamente correlacionadas (> threshold).
    Devuelve un nuevo DataFrame con features seleccionadas.
    """
    corr = X.corr().abs()
    upper = corr.where(pd.np.triu(pd.np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X.drop(columns=to_drop), to_drop
