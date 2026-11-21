"""Test script for Phase 4 preprocessing."""
import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.preprocessing import prepare_data, build_preprocessor

def main():
    print("=== Testing Preprocessing Pipeline ===")
    X_train, X_test, y_train, y_test, label_encoder = prepare_data()
    
    print("\n=== Data Shapes ===")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")
    
    print("\n=== Target Classes ===")
    print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    
    print("\n=== Fitting Preprocessor on Train Data ===")
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed X_train shape: {X_train_processed.shape}")
    print(f"Processed X_test shape:  {X_test_processed.shape}")
    
    # Verify no NaNs remain
    import numpy as np
    if np.isnan(X_train_processed).any():
        print("[ERROR] NaNs found in processed training data!")
    else:
        print("[SUCCESS] No NaNs found in processed data.")

if __name__ == "__main__":
    main()