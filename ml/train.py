"""Train, compare, and track ML models using MLflow."""
import time
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ml.preprocessing import prepare_data, build_preprocessor
from ml import config

ARTIFACTS_DIR = Path("artifacts/model")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def get_models():
    """Define models with class_weight='balanced' to handle imbalance."""
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, n_jobs=-1, random_state=42
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=15, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", n_estimators=100, max_depth=15, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, 
            eval_metric="mlogloss", n_jobs=-1, random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, 
            class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1
        )
    }

def train_and_evaluate():
    print("=== Phase 6/7: Model Training with MLflow Tracking ===")
    
    # 1. Setup MLflow (menggunakan local file store jika server tidak tersedia)
    # Anda bisa mengganti URI ini dengan http://localhost:5000 jika menjalankan mlflow server
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("RoadSafe-Severity-Classification")
    
    # 2. Load Data
    X_train, X_test, y_train, y_test, le = prepare_data()
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    models = get_models()
    
    best_model_name = None
    best_f1_macro = -1
    best_pipeline = None

    # 3. Loop through models and track with MLflow
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        with mlflow.start_run(run_name=name) as run:
            start_time = time.time()
            
            pipeline = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", model)
            ])
            
            fit_params = {}
            if name == "XGBoost":
                fit_params["classifier__sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_train)

            scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "f1_weighted": "f1_weighted"}
            
            cv_results = cross_validate(
                pipeline, X_train, y_train, 
                cv=cv, scoring=scoring, 
                fit_params=fit_params, n_jobs=1
            )
            
            mean_f1_macro = cv_results["test_f1_macro"].mean()
            mean_f1_weighted = cv_results["test_f1_weighted"].mean()
            mean_acc = cv_results["test_accuracy"].mean()
            elapsed = time.time() - start_time
            
            print(f"Time: {elapsed:.1f}s | Acc: {mean_acc:.4f} | F1-Macro: {mean_f1_macro:.4f}")

            # Log parameters & metrics to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metric("cv_accuracy", mean_acc)
            mlflow.log_metric("cv_f1_macro", mean_f1_macro)
            mlflow.log_metric("cv_f1_weighted", mean_f1_weighted)
            mlflow.log_metric("training_time_seconds", elapsed)
            
            # Check if this is the best model
            if mean_f1_macro > best_f1_macro:
                best_f1_macro = mean_f1_macro
                best_model_name = name
                
                # Fit on full train set
                print(f"New best model! Fitting {name} on full training set...")
                final_pipeline = Pipeline([
                    ("preprocessor", build_preprocessor()),
                    ("classifier", models[name])
                ])
                final_pipeline.fit(X_train, y_train, **fit_params)
                best_pipeline = final_pipeline
                
                # Log model to MLflow
                mlflow.sklearn.log_model(
                    sk_model=final_pipeline,
                    artifact_path="model",
                    registered_model_name="RoadSafe-AI-Model"
                )

    # 4. Save Locally as well (for FastAPI fallback)
    print(f"\n=== Winning Model: {best_model_name} (F1-Macro: {best_f1_macro:.4f}) ===")
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    joblib.dump(best_pipeline, model_path)
    print(f"[saved] Best model saved locally to {model_path}")
    
    metadata = {
        "best_model": best_model_name,
        "best_f1_macro": float(best_f1_macro),
        "label_encoder_classes": list(le.classes_)
    }
    with open(ARTIFACTS_DIR / "training_results.json", "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    train_and_evaluate()