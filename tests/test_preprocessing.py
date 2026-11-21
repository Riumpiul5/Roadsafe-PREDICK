"""Test ML preprocessing functions."""
import pandas as pd
import pytest
from ml.preprocessing import clean_data, engineer_features
from ml import config

def test_clean_data_removes_duplicates():
    df = pd.DataFrame({
        "collision_index": [1, 1, 2],
        config.TARGET_COL: [1, 1, 2]
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 2
    assert cleaned["collision_index"].tolist() == [1, 2]

def test_clean_data_handles_unknown_values():
    df = pd.DataFrame({
        "collision_index": [1],
        config.TARGET_COL: [1],
        "weather_conditions": [-1], # Kode unknown STATS19
        "speed_limit": [-1]
    })
    cleaned = clean_data(df)
    # -1 di kategorik harus jadi string "Unknown"
    assert cleaned["weather_conditions"].iloc[0] == "Unknown"
    # -1 di numerik harus jadi NaN
    assert pd.isna(cleaned["speed_limit"].iloc[0])

def test_engineer_features():
    df = pd.DataFrame({
        "date": ["2023-11-21"],
        "time": ["18:30"]
    })
    df = engineer_features(df)
    assert df["year"].iloc[0] == 2023
    assert df["month"].iloc[0] == 11
    assert df["day_of_week"].iloc[0] == 1 # Selasa
    assert df["hour"].iloc[0] == 18