"""Generate a dummy model artifact for CI/CD testing."""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml import config


def create_dummy_model():
    print("Creating dummy model for CI...")
    
    # Data dummy (1 baris) agar pipeline bisa di-fit
    df = pd.DataFrame([{
        "longitude": -0.123, "latitude": 51.507, "speed_limit": 30,
        "hour": 18, "day_of_week": 5, "month": 11,
        "road_type": "Single carriageway", "weather_conditions": "Raining",
        "road_surface_conditions": "Wet", "light_conditions": "Darkness - lights lit",
        "urban_or_rural_area": "Urban", "junction_detail": "T or staggered junction"
    }])
    y = [2] # Dummy target (Slight)

    num_cols = config.NUMERICAL_FEATURES
    cat_cols = config.CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
    ])

    pipeline.fit(df, y)
    
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Dummy model saved to {model_path}")

if __name__ == "__main__":
    create_dummy_model()