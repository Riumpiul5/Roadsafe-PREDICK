"""Generate global SHAP explanations and save plots."""
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from ml.preprocessing import prepare_data
from ml import config

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def generate_global_shap():
    print("Loading data and model...")
    X_train, X_test, y_train, y_test, le = prepare_data()
    
    # Ambil sampel 500 baris dari test set
    X_sample = X_test.sample(500, random_state=42)
    
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    pipeline = joblib.load(model_path)
    
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    X_transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()
    
    print("Calculating SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)
    
    # --- Plot 1: Global Feature Importance (Bar Chart) ---
    # Cukup berikan list/array shap_values langsung. SHAP akan otomatis menghitung
    # rata-rata absolutnya untuk semua kelas.
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X_transformed, 
        feature_names=feature_names, 
        plot_type="bar", 
        class_names=list(le.classes_),
        show=False
    )
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_shap_global_importance.png", bbox_inches="tight")
    plt.close()
    print(f"[saved] Global SHAP plot saved to {FIG_DIR}/08_shap_global_importance.png")
    
    # --- Plot 2: SHAP Beeswarm (Kelas Fatal / Index 0) ---
    # Ambil nilai SHAP khusus untuk kelas Fatal (index 0)
    if isinstance(shap_values, list):
        sv_class0 = shap_values[0]
    elif len(shap_values.shape) == 3: # XGBoost versi baru
        sv_class0 = shap_values[:, :, 0]
    else:
        sv_class0 = shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv_class0, 
        X_transformed, 
        feature_names=feature_names, 
        show=False
    )
    plt.title("SHAP Summary Plot (Class: Fatal)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_shap_summary_beeswarm.png", bbox_inches="tight")
    plt.close()
    print(f"[saved] SHAP Beeswarm plot saved to {FIG_DIR}/09_shap_summary_beeswarm.png")

if __name__ == "__main__":
    generate_global_shap()