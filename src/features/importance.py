import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def feature_importance(X: pd.DataFrame, y: pd.Series, top_k: int = 7):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected = imp.head(top_k).index.tolist()
    return imp, selected
