import pandas as pd
import numpy as np

# ---------------- comunes ----------------
def _fit_stats(df: pd.DataFrame, num_cols):
    return {c: (df[c].min(), df[c].max()) for c in num_cols if c in df.columns}

def _apply_scale(df, stats, num_cols):
    df = df.copy()
    for c in num_cols:
        if c in df.columns and c in stats:
            mn, mx = stats[c]
            df[c + "_scaled"] = (df[c] - mn) / (mx - mn + 1e-8)
    return df

def _align_and_concat(X: pd.DataFrame, y: pd.Series, feature_cols):
    X = X.reindex(columns=feature_cols, fill_value=0.0).astype("float32")
    return pd.concat([X, y.rename("target")], axis=1)


def preprocess_split_dfs(train_df: pd.DataFrame,
                         val_df: pd.DataFrame,
                         test_df: pd.DataFrame):
    """
    Preprocesamiento para Spaceship Titanic SIN dummies.
    Todo termina en columnas numéricas (int/float).
    Devuelve: train_p, val_p, test_p, feature_cols
    """

    # 0) Normalizar nombres
    def _norm_cols(d):
        d = d.copy()
        d.columns = d.columns.str.strip()
        return d

    train_df = _norm_cols(train_df)
    val_df   = _norm_cols(val_df)
    test_df  = _norm_cols(test_df)

    # 1) Transported -> target
    def map_target(df):
        y = df["Transported"].map({
            True: 1, "True": 1, 1: 1,
            False: 0, "False": 0, 0: 0
        }).astype(int)
        df = df.drop(columns=["Transported"])
        df["target"] = y
        return df

    train_df = map_target(train_df)
    val_df   = map_target(val_df)
    test_df  = map_target(test_df)

    # 2) Preprocesamiento base en TRAIN para stats
    train = train_df.copy()

    # Age mediana
    train["Age"] = train["Age"].fillna(train["Age"].median())

    # Booleans -> 0/1
    for c in ["VIP", "CryoSleep"]:
        train[c] = train[c].map({
            True: 1, "True": 1, 1: 1,
            False: 0, "False": 0, 0: 0
        })
        train[c] = train[c].fillna(0).astype(int)

    # HomePlanet / Destination
    train["HomePlanet"]  = train["HomePlanet"].fillna("Unknown")
    train["Destination"] = train["Destination"].fillna("Unknown")

    # Drop name/id
    train = train.drop(columns=[c for c in ["Name", "PassengerId"] if c in train.columns],
                       errors="ignore")

    # Cabin -> Deck / CabinNum / Side
    parts = train["Cabin"].astype(str).str.split("/", expand=True)
    train["Deck"]     = parts[0].replace({"nan": np.nan})
    train["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
    train["Side"]     = parts[2].replace({"nan": np.nan})

    # Stats de Deck/CabinNum
    deck_mode = train["Deck"].mode().iloc[0] if not train["Deck"].mode().empty else "Unknown"
    side_mode = train["Side"].mode().iloc[0] if not train["Side"].mode().empty else "U"
    deck_median_num = train.groupby("Deck")["CabinNum"].median()
    global_median_num = train["CabinNum"].median()

    # Guardamos train limpio como referencia
    train_ref = train.copy()

    # 3) Limpieza que usa stats de TRAIN
    def basic_clean(df):
        df = df.copy()

        # Age con mediana de train
        df["Age"] = df["Age"].fillna(train_ref["Age"].median())

        # Booleans -> 0/1
        for c in ["VIP", "CryoSleep"]:
            df[c] = df[c].map({
                True: 1, "True": 1, 1: 1,
                False: 0, "False": 0, 0: 0
            })
            df[c] = df[c].fillna(0).astype(int)

        # HomePlanet / Destination
        df["HomePlanet"]  = df["HomePlanet"].fillna("Unknown")
        df["Destination"] = df["Destination"].fillna("Unknown")

        # Eliminar columnas irrelevantes
        df = df.drop(columns=[c for c in ["Name", "PassengerId"] if c in df.columns],
                     errors="ignore")

        # Cabin -> Deck/CabinNum/Side
        parts = df["Cabin"].astype(str).str.split("/", expand=True)
        df["Deck"]     = parts[0].replace({"nan": np.nan})
        df["CabinNum"] = pd.to_numeric(parts[1], errors="coerce")
        df["Side"]     = parts[2].replace({"nan": np.nan})

        # Imputación usando stats de train_ref
        df["Deck"] = df["Deck"].fillna(deck_mode)
        df["Side"] = df["Side"].fillna(side_mode)
        df["CabinNum"] = df.apply(
            lambda r: deck_median_num.get(r["Deck"], global_median_num)
                      if pd.isna(r["CabinNum"]) else r["CabinNum"],
            axis=1
        )

        return df

    # Aplicar limpieza a los tres splits
    train_df = basic_clean(train_df)
    val_df   = basic_clean(val_df)
    test_df  = basic_clean(test_df)

    # 4) Crear features (tu lógica)
    def add_feats(df, stats):
        df = df.copy()
        # IsSenior
        df["IsSenior"] = df["Age"].apply(lambda x: 1 if x >= 55 else 0)

        # Escalar numéricos
        df = _apply_scale(df, stats, ["Age", "RoomService", "FoodCourt",
                                      "ShoppingMall", "Spa", "VRDeck"])

        # Bills
        df["Bills"] = (
            df.get("RoomService_scaled", 0)
            + df.get("FoodCourt_scaled", 0)
            + df.get("ShoppingMall_scaled", 0)
            + df.get("Spa_scaled", 0)
            + df.get("VRDeck_scaled", 0)
        )

        # Destination numérica como en tu ejemplo
        if "Destination" in df.columns and df["Destination"].dtype == object:
            df["Destination"] = df["Destination"].replace({
                "TRAPPIST-1e": 0,
                "TRAPPIST- 1e": 0,
                "55 Cancri e": 1,
                "PSO J318.5-22": 2,
                "Unknown": -1
            })

        return df

    # Stats de escalado SOLO de train_ref
    scale_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    stats = _fit_stats(train_ref, scale_cols)

    train_df = add_feats(train_df, stats)
    val_df   = add_feats(val_df,   stats)
    test_df  = add_feats(test_df,  stats)

    # 5) Codificar HomePlanet, Deck, Side a ENTEROS (SIN dummies)
    home_map = {
        "Earth": 0,
        "Europa": 1,
        "Mars": 2,
        "Unknown": -1
    }
    deck_map = {
        "A": 0, "B": 1, "C": 2, "D": 3,
        "E": 4, "F": 5, "G": 6, "T": 7,
        "Unknown": -1
    }
    side_map = {
        "P": 0,
        "S": 1,
        "U": -1
    }

    def encode_cats(df):
        df = df.copy()
        if "HomePlanet" in df.columns:
            df["HomePlanet"] = df["HomePlanet"].map(home_map).fillna(-1).astype(int)
        if "Deck" in df.columns:
            df["Deck"] = df["Deck"].map(deck_map).fillna(-1).astype(int)
        if "Side" in df.columns:
            df["Side"] = df["Side"].map(side_map).fillna(-1).astype(int)
        return df

    train_df = encode_cats(train_df)
    val_df   = encode_cats(val_df)
    test_df  = encode_cats(test_df)

    # 6) Selección final de columnas tipo tu df_train
    def select_final(df):
        cols = [
            "HomePlanet",
            "CryoSleep",
            "Age_scaled",
            "Deck",
            "CabinNum",
            "Side",
            "Bills",
            "RoomService_scaled",
            "FoodCourt_scaled",
            "Spa_scaled",
            "VRDeck_scaled",
            "target"
        ]
        existing = [c for c in cols if c in df.columns]
        return df[existing].copy()

    train_df = select_final(train_df)
    val_df   = select_final(val_df)
    test_df  = select_final(test_df)

    # 7) SIN dummies: solo separamos X,y y alineamos columnas
    X_tr = train_df.drop(columns=["target"])
    y_tr = train_df["target"].astype(int)
    feature_cols = list(X_tr.columns)

    X_v  = val_df.drop(columns=["target"])
    y_v  = val_df["target"].astype(int)

    X_te = test_df.drop(columns=["target"])
    y_te = test_df["target"].astype(int)

    train_p = _align_and_concat(X_tr,  y_tr,  feature_cols)
    val_p   = _align_and_concat(X_v,   y_v,   feature_cols)
    test_p  = _align_and_concat(X_te,  y_te,  feature_cols)

    return train_p, val_p, test_p, feature_cols
