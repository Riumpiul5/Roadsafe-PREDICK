"""Data preprocessing pipeline for RoadSafe AI.

Handles schema normalization, cleaning, feature engineering,
and building a scikit-learn ColumnTransformer.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from ml import config


def load_and_normalize(path: str = None) -> pd.DataFrame:
    """Load raw data and normalize legacy schema to modern schema."""
    path = path or config.RAW_DATA_PATH
    df = pd.read_csv(path, low_memory=False)

    rename_map = {
        "accident_index": "collision_index",
        "accident_severity": "collision_severity",
        "accident_year": "collision_year",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data: drop duplicates, handle STATS19 missing codes, drop target nulls."""
    if "collision_index" in df.columns:
        df = df.drop_duplicates(subset=["collision_index"]).copy()

    df = df.dropna(subset=[config.TARGET_COL]).copy()

    # 1. NUMERICAL: Replace -1 (Unknown) with NaN
    for col in config.NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)

    # 2. CATEGORICAL: Replace -1 with NaN, convert to string, and fill with "Unknown"
    # This prevents the float64 vs str ValueError in SimpleImputer
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)
            # astype(str) converts 1.0 -> '1.0', then we fill NaN with "Unknown"
            df[col] = df[col].fillna("Unknown").astype(str)

    # 3. Filter out invalid coordinates (0.0 means missing in some STATS19 files)
    if "longitude" in df.columns and "latitude" in df.columns:
        df = df[(df["longitude"] != 0.0) & (df["latitude"] != 0.0)]
        df = df[(df["longitude"] > -8) & (df["longitude"] < 2) & 
                (df["latitude"] > 50) & (df["latitude"] < 59)]

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from 'date' and 'time' columns."""
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek

    if "time" in df.columns:
        df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour

    return df


def get_train_test_split(df: pd.DataFrame, test_size_years: int = 1):
    """Temporal split: train on older years, test on the most recent year(s)."""
    max_year = df["year"].max()
    test_year_threshold = max_year - test_size_years + 1

    train_df = df[df["year"] < test_year_threshold].copy()
    test_df = df[df["year"] >= test_year_threshold].copy()

    print(f"[split] Max year: {max_year}")
    print(f"[split] Train: {len(train_df)} rows (Years < {test_year_threshold})")
    print(f"[split] Test:  {len(test_df)} rows (Years >= {test_year_threshold})")

    return train_df, test_df


def build_preprocessor() -> ColumnTransformer:
    """Build the sklearn ColumnTransformer."""
    
    num_features = config.NUMERICAL_FEATURES.copy()
    if not config.USE_COORDINATES:
        num_features = [c for c in num_features if c not in ["longitude", "latitude"]]

    # Numerical pipeline: Impute with median (handles NaNs from -1 and missing coords), then scale
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline: One-hot encode (no imputer needed because we already handled it in clean_data)
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_data():
    """End-to-end data preparation."""
    df = load_and_normalize()
    df = clean_data(df)
    df = engineer_features(df)
    
    train_df, test_df = get_train_test_split(df)

    # --- PERBAIKAN: Map angka STATS19 ke Teks sebelum di-encode ---
    # Jika datanya masih angka (1, 2, 3), ubah ke teks (Fatal, Serious, Slight)
    if train_df[config.TARGET_COL].dtype != object:
        train_df[config.TARGET_COL] = train_df[config.TARGET_COL].map(config.SEVERITY_MAPPING)
        test_df[config.TARGET_COL] = test_df[config.TARGET_COL].map(config.SEVERITY_MAPPING)

    # Sekarang LabelEncoder akan menghasilkan classes_ = ['Fatal', 'Serious', 'Slight']
    le = LabelEncoder()
    train_df[config.TARGET_COL] = le.fit_transform(train_df[config.TARGET_COL])
    test_df[config.TARGET_COL] = le.transform(test_df[config.TARGET_COL])

    X_train = train_df.drop(columns=[c for c in config.DROP_COLUMNS if c in train_df.columns] + [config.TARGET_COL])
    y_train = train_df[config.TARGET_COL]
    
    X_test = test_df.drop(columns=[c for c in config.DROP_COLUMNS if c in test_df.columns] + [config.TARGET_COL])
    y_test = test_df[config.TARGET_COL]

    return X_train, X_test, y_train, y_test, le