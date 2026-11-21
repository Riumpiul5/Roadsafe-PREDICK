"""Configuration constants for the ML pipeline."""
from pathlib import Path

# --- Paths ---
RAW_DATA_PATH = Path("dataset/raw/collision_last_5_years.csv")
PROCESSED_DIR = Path("dataset/processed")
ARTIFACTS_DIR = Path("artifacts/model")

# Tambahan variabel untuk kompatibilitas dengan train.py, evaluate.py, dan predict.py
MODEL_DIR = ARTIFACTS_DIR
MODEL_FILENAME = "roadsafe_pipeline.joblib"

# --- Feature Configuration ---
TARGET_COL = "collision_severity"

# Numerical features used in the model
NUMERICAL_FEATURES = [
    "longitude",
    "latitude",
    "speed_limit",
    "hour",
    "day_of_week",
    "month",
]

# Categorical features used in the model
CATEGORICAL_FEATURES = [
    "road_type",
    "weather_conditions",
    "road_surface_conditions",
    "light_conditions",
    "urban_or_rural_area",
    "junction_detail",
]

# Columns that MUST be dropped to prevent data leakage or because they are IDs
DROP_COLUMNS = [
    "collision_index",       # ID
    "collision_year",        # Used for temporal split, not as feature
    "date",                  # Used for feature engineering, then dropped
    "time",                  # Used for feature engineering, then dropped
    "casualty_class",        # Leakage
    "casualty_severity",     # Leakage (defines target)
    "number_of_casualties",  # Leakage
    "number_of_vehicles",    # Leakage
    "accident_index",        # Legacy ID
    "accident_severity",     # Legacy target
]

# Label mapping (STATS19 standard: 1=Fatal, 2=Serious, 3=Slight)
SEVERITY_MAPPING = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}

# Flag for coordinate experiment (Phase 5)
USE_COORDINATES = True