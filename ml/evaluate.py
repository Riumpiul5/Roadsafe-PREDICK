"""Evaluate the saved RoadSafe AI model on the holdout test set."""
import joblib
from sklearn.metrics import classification_report, confusion_matrix

from ml import config
from ml.preprocessing import prepare_data


def evaluate_model():
    print("=== Phase 5: Model Evaluation on Test Set ===")
    
    # 1. Load Data
    _, X_test, _, y_test, le = prepare_data()
    
    # 2. Load Model
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    print(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)
    
    # 3. Predict
    print("Predicting on test set...")
    y_pred = pipeline.predict(X_test)
    
    # 4. Print Metrics
    target_names = le.classes_
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'':>10} | {' '.join([f'{c:>8}' for c in target_names])}")
    print("-" * 50)
    for i, row in enumerate(cm):
        print(f"{target_names[i]:>10} | {' '.join([f'{val:>8}' for val in row])}")

if __name__ == "__main__":
    evaluate_model()