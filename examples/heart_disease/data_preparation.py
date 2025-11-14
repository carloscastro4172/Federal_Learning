import pandas as pd
import numpy as np

def _fit_stats(df, num_cols):
    return {c: (df[c].min(), df[c].max()) for c in num_cols}

def _apply_scale(df, stats, num_cols):
    df = df.copy()
    for c in num_cols:
        mn, mx = stats[c]
        df[c + "_scaled"] = (df[c] - mn) / (mx - mn + 1e-8)
    return df.drop(columns=num_cols, errors="ignore")

def preprocess_split_dfs(train_df: pd.DataFrame,
                         val_df: pd.DataFrame,
                         test_df: pd.DataFrame):
    """
    Devuelve (train_df_p, val_df_p, test_df_p, feature_cols)
    con target='target' y features alineadas entre splits.
    """
    train_df = train_df.copy(); train_df.columns = train_df.columns.str.strip()
    val_df   = val_df.copy();   val_df.columns   = val_df.columns.str.strip()
    test_df  = test_df.copy();  test_df.columns  = test_df.columns.str.strip()
    assert "target" in train_df.columns, "No existe la columna 'target' en train"

    num_cols = ["resting bp s", "cholesterol"]
    stats = _fit_stats(train_df, num_cols)

    def transform(d: pd.DataFrame):
        d = d.copy()
        y = d["target"].astype(int)
        X = d.drop(columns=["target"])
        X = _apply_scale(X, stats, num_cols)
        X = pd.get_dummies(
            X,
            columns=["sex","chest pain type","fasting blood sugar",
                     "resting ecg","exercise angina","ST slope"],
            drop_first=True
        )
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
        return X, y

    X_tr, y_tr = transform(train_df)
    feature_cols = list(X_tr.columns)

    def align(X, y):
        X = X.reindex(columns=feature_cols, fill_value=0.0).astype("float32")
        return pd.concat([X, y.rename("target")], axis=1)

    X_v,  y_v  = transform(val_df)
    X_te, y_te = transform(test_df)

    train_p = align(X_tr, y_tr)
    val_p   = align(X_v,  y_v)
    test_p  = align(X_te, y_te)

    return train_p, val_p, test_p, feature_cols
