import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# --- Config ---
input_folder = "Processed_Data"
output_folder = "Model_Input"
os.makedirs(output_folder, exist_ok=True)

SPLIT_DATE = '2022-05-01'
MARKETS = ["nifty50", "eurostoxx50"]

def split_scale(market_name):
    print(f"\nSPLITTING AND SCALING: {market_name.upper()}")
    
    # 1. Load Data
    input_file = os.path.join(input_folder, f"{market_name}_processed_unscaled.csv")
    
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found. Run Preprocess.py first.")
        return
    
    df = pd.read_csv(input_file, index_col='Date', parse_dates=True)
    print(f"Loaded {len(df)} rows.")
    
    # 2. Define Features (X) and Target (y)
    features_to_drop = ['Label', 'Ticker']
    features = [col for col in df.columns if col not in features_to_drop]
    
    print(f"Features identified: {len(features)}")
    
    X = df[features]
    y = df['Label']

    # 3. Time-Based Train-Test Split
    train_mask = X.index < SPLIT_DATE
    test_mask = X.index >= SPLIT_DATE

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"Train set: {X_train.shape[0]} rows")
    print(f"Test set:  {X_test.shape[0]} rows")
    
    # 4. Scale Data
    print("Scaling features...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Save Files
    print("Saving model-ready files...")
    np.save(os.path.join(output_folder, f"{market_name}_X_train.npy"), X_train_scaled)
    np.save(os.path.join(output_folder, f"{market_name}_X_test.npy"), X_test_scaled)
    np.save(os.path.join(output_folder, f"{market_name}_y_train.npy"), y_train.values)
    np.save(os.path.join(output_folder, f"{market_name}_y_test.npy"), y_test.values)
    
    feature_names_path = os.path.join(output_folder, f"{market_name}_feature_names.csv")
    pd.Series(features).to_csv(feature_names_path, index=False, header=False)
    
    print(f"Saved all files for {market_name}.")

def main():
    print("TRAIN-TEST SPLIT & SCALING PIPELINE")
    
    for market in MARKETS:
        split_scale(market)
    
    print("\nSPLIT & SCALING COMPLETE!")

if __name__ == "__main__":
    main()