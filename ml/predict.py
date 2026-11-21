"""Inference module for RoadSafe AI."""
import pandas as pd
import joblib
from pathlib import Path
from ml import config

class SeverityPredictor:
    def __init__(self, model_path: Path = None):
        self.model_path = model_path or (config.MODEL_DIR / config.MODEL_FILENAME)
        self.pipeline = None
        self.classes_ = ["Fatal", "Serious", "Slight"] # Sesuai LabelEncoder di Phase 4

    def load(self):
        """Load the model pipeline from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        return self

    def predict_single(self, input_dict: dict) -> dict:
        """Predict a single accident record."""
        if self.pipeline is None:
            self.load()
            
        # Convert dict to DataFrame
        df = pd.DataFrame([input_dict])
        
        # Predict probability
        probs = self.pipeline.predict_proba(df)[0]
        pred_idx = self.pipeline.predict(df)[0]
        
        return {
            "prediction": self.classes_[pred_idx],
            "probabilities": {
                self.classes_[i]: float(probs[i]) for i in range(len(self.classes_))
            }
        }

# Contoh penggunaan jika dijalankan langsung
if __name__ == "__main__":
    predictor = SeverityPredictor().load()
    
    sample_input = {
        "longitude": -0.123,
        "latitude": 51.507,
        "speed_limit": 30,
        "hour": 18,
        "day_of_week": 5,
        "month": 11,
        "road_type": "Single carriageway",
        "weather_conditions": "Raining",
        "road_surface_conditions": "Wet",
        "light_conditions": "Darkness - lights lit",
        "urban_or_rural_area": "Urban",
        "junction_detail": "T or staggered junction"
    }
    
    # Note: The input data must be preprocessed exactly like training data.
    # If using raw strings from API, the model expects mapped codes or the API
    # must map strings to the exact categorical levels in the training data.
    # For this simple test, we'll just show it runs.
    print("Test prediction structure ready.")