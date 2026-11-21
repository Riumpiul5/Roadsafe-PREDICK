"""Model service for loading and predicting with the RoadSafe AI model."""
import logging
import uuid
import shap
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

import joblib
import pandas as pd
from app.core.config import get_settings
from app.schemas.prediction import PredictionRequest

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.settings = get_settings()
        self.pipeline = None
        self.model_version = self.settings.model_version
        # Label mapping sesuai LabelEncoder di Phase 4 (Fatal=0, Serious=1, Slight=2)
        # Dibuat eksplisit agar mapping probabilitas tidak tertukar
        self.classes_ = ["Fatal", "Serious", "Slight"]

    def load_model(self):
        """Load the joblib pipeline from disk."""
        path = self.settings.model_path
        if not path.exists():
            logger.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"Model not found at {path}")
        
        self.pipeline = joblib.load(path)
        logger.info(f"Model loaded successfully from {path}")

    def predict(self, request: PredictionRequest, request_id: str = None) -> Dict[str, Any]:
        """Predict a single record."""
        if self.pipeline is None:
            raise RuntimeError("Model is not loaded.")
        
        request_id = request_id or str(uuid.uuid4())
        
        # Convert Pydantic model to DataFrame (expected by the sklearn pipeline)
        input_dict = request.model_dump()
        df = pd.DataFrame([input_dict])
        
        try:
            probs = self.pipeline.predict_proba(df)[0]
            pred_idx = self.pipeline.predict(df)[0]
            
            return {
                "prediction": self.classes_[pred_idx],
                "probabilities": {
                    self.classes_[i]: round(float(probs[i]), 4) for i in range(len(self.classes_))
                },
                "model_version": self.model_version,
                "request_id": request_id
            }
        except Exception as e:
            logger.error(f"Prediction failed for request {request_id}: {e}")
            raise e

    def predict_batch(self, requests: List[PredictionRequest]) -> List[Dict[str, Any]]:
        """Predict a batch of records."""
        return [self.predict(req) for req in requests]
    
    def explain_prediction(self, request: PredictionRequest, request_id: str = None) -> Dict[str, Any]:
        """Predict and explain using SHAP."""
        if self.pipeline is None:
            raise RuntimeError("Model is not loaded.")
        
        request_id = request_id or str(uuid.uuid4())
        
        # 1. Dapatkan prediksi probabilitas seperti biasa
        base_result = self.predict(request, request_id)
        
        # 2. Siapkan data
        df = pd.DataFrame([request.model_dump()])
        preprocessor = self.pipeline.named_steps["preprocessor"]
        classifier = self.pipeline.named_steps["classifier"]
        
        X_transformed = preprocessor.transform(df)
        feature_names = preprocessor.get_feature_names_out()
        
        # 3. Hitung SHAP values
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
        
        # Dapatkan index kelas yang diprediksi
        pred_class_name = base_result["prediction"]
        pred_idx = self.classes_.index(pred_class_name)
        
        # Ambil SHAP values untuk kelas yang diprediksi
        if isinstance(shap_values, list):
            class_shap_values = shap_values[pred_idx][0] # shape: (n_features,)
        else:
            # Beberapa versi XGBoost mengembalikan shape (n_samples, n_features, n_classes)
            if len(shap_values.shape) == 3:
                class_shap_values = shap_values[0, :, pred_idx]
            else:
                class_shap_values = shap_values[0]
                
        # 4. Format Feature Contributions
        # Bersihkan nama fitur (misal: "cat__weather_conditions_Raining" -> "weather_conditions_Raining")
        contributions = []
        for i, (name, val) in enumerate(zip(feature_names, class_shap_values)):
            if abs(val) > 0.01: # Hanya tampilkan fitur yang pengaruhnya signifikan
                clean_name = name.replace("num__", "").replace("cat__", "")
                contributions.append({
                    "feature": clean_name,
                    "value": round(float(val), 4),
                    "impact": "increases" if val > 0 else "decreases"
                })
                
        # Urutkan berdasarkan nilai absolut terbesar
        contributions = sorted(contributions, key=lambda x: abs(x["value"]), reverse=True)[:5]
        
        base_result["explanation"] = contributions
        return base_result

# Singleton instance
model_service = ModelService()