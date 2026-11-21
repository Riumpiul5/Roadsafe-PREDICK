"""Generate a dummy model artifact for CI/CD testing."""
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from pathlib import Path
from ml import config

def create_dummy_model():
    print("Creating dummy model for CI...")
    
    # Data dummy (2 baris dengan kelas berbeda) agar model bisa di-fit
    df = pd.DataFrame([
        {
            "longitude": -0.123, "latitude": 51.507, "speed_limit": 30,
            "hour": 18, "day_of_week": 5, "month": 11,
            "road_type": "Single carriageway", "weather_conditions": "Raining",
            "road_surface_conditions": "Wet", "light_conditions": "Darkness - lights lit",
            "urban_or_rural_area": "Urban", "junction_detail": "T or staggered junction"
        },
        {
            "longitude": -0.500, "latitude": 51.600, "speed_limit": 60,
            "hour": 14, "day_of_week": 2, "month": 6,
            "road_type": "Dual carriageway", "weather_conditions": "Fine no high winds",
            "road_surface_conditions": "Dry", "light_conditions": "Daylight",
            "urban_or_rural_area": "Rural", "junction_detail": "Not at junction or within 20 metres"
        }
    ])
    
    # Target dummy: 2 (Slight) dan 0 (Fatal)
    y = [2, 0] 

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