import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import os
import joblib

# --- Config ---
input_folder = "Model_Input"
output_folder = "Model_Output"
os.makedirs(output_folder, exist_ok=True)

MARKETS_TO_TRAIN = ["nifty50", "eurostoxx50"]
N_JOBS = -1 

def train_model(market_name):
    print(f"\nTRAINING MODEL FOR {market_name.upper()}")
    
    # 1. Load Data
    print("Loading training data...")
    try:
        X_train = np.load(os.path.join(input_folder, f"{market_name}_X_train.npy"))
        X_test = np.load(os.path.join(input_folder, f"{market_name}_X_test.npy"))
        y_train = np.load(os.path.join(input_folder, f"{market_name}_y_train.npy"))
        y_test = np.load(os.path.join(input_folder, f"{market_name}_y_test.npy"))
    except FileNotFoundError:
        print(f"ERROR: Data files for {market_name} not found. Run Train_Test_Split.py first.")
        return

    # 2. Check Class Balance
    class_balance_ratio = np.mean(y_train)
    print(f"Training Class 'Buy' (1): {class_balance_ratio:.2%}")

    # 3. Define Model and Parameter Grid
    print("Starting hyperparameter tuning...")
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        
        enable_categorical=False,
        random_state=42
    )
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [200, 400],
        'scale_pos_weight': [2, 3.5, 5]  # To handle class imbalance
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=tscv,
        n_jobs=N_JOBS,
        verbose=1 
    )
    
    # 4. Run Tuning
    grid_search.fit(X_train, y_train)
    
    print("\nTUNING COMPLETE")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_

    # 5. Evaluate on Test Set
    print("\nTEST SET EVALUATION")
    preds_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y_test, preds_prob)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    
    best_threshold = 0.5
    if len(f1_scores) > 0:
        best_threshold = thresholds[np.argmax(f1_scores)]

    preds_binary_best = (preds_prob >= best_threshold).astype(int)
    
    print(f"Using Optimal Threshold: {best_threshold:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds_binary_best, target_names=['Hold', 'Buy']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds_binary_best))

    # 6. Save Model
    model_filename = os.path.join(output_folder, f"{market_name}_best_model.joblib")
    joblib.dump(best_model, model_filename)
    print(f"\nModel saved to: {model_filename}")

if __name__ == "__main__":
    for market in MARKETS_TO_TRAIN:
        train_model(market)
    print("\nALL MODEL TRAINING COMPLETE!")