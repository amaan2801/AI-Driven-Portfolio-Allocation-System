import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# --- Config ---
INPUT_FOLDER_MODEL_DATA = "Model_Input"
INPUT_FOLDER_MODEL_OUTPUT = "Model_Output"
MARKETS_TO_RUN = ["nifty50", "eurostoxx50"]

def main():
    for market_name in MARKETS_TO_RUN:
        print(f"\nGENERATING FEATURE IMPORTANCE FOR {market_name.upper()} MODEL")
        
        # 1. Load Model
        model_path = os.path.join(INPUT_FOLDER_MODEL_OUTPUT, f"{market_name}_best_model.joblib")
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"ERROR: Model file not found at {model_path}. Run Model_Train.py first.")
            continue
        
        # 2. Load Data
        X_train_path = os.path.join(INPUT_FOLDER_MODEL_DATA, f"{market_name}_X_train.npy")
        y_train_path = os.path.join(INPUT_FOLDER_MODEL_DATA, f"{market_name}_y_train.npy")
        try:
            X_train = np.load(X_train_path)
            y_train = np.load(y_train_path)
        except FileNotFoundError:
            print(f"ERROR: Data files for {market_name} not found. Run Train_Test_Split.py first.")
            continue
            
        # 3. Load Feature Names
        features_path = os.path.join(INPUT_FOLDER_MODEL_DATA, f"{market_name}_feature_names.csv")
        feature_names = pd.read_csv(features_path, header=None).iloc[:, 0].tolist()
        
        if X_train.shape[1] != len(feature_names):
            print(f"ERROR: Dimension mismatch! {X_train.shape[1]} vs {len(feature_names)}")
            continue
        
        # 4. Method 1: XGBoost Built-in Importance
        print("Calculating XGBoost feature importance (Gain)...")
        importance_gain = model.get_booster().get_score(importance_type='gain')
        importance_dict = {}
        for key, value in importance_gain.items():
            feature_idx = int(key[1:])
            importance_dict[feature_names[feature_idx]] = value
        
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance_gain': list(importance_dict.values())
        }).sort_values('importance_gain', ascending=False)
        
        # 5. Method 2: Permutation Importance
        print("Calculating Permutation importance... (this may take a minute)")
        sample_size = min(5000, len(X_train))
        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
        
        perm_importance = permutation_importance(
            model, 
            X_train[sample_indices], 
            y_train[sample_indices], 
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        perm_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_perm': perm_importance.importances_mean
        }).sort_values('importance_perm', ascending=False)
        
        # 6. Generate Visualization
        print("Generating feature importance plot...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f"Feature Importance for {market_name.upper()}", fontsize=16, fontweight='bold')
        
        top_gain = importance_df.head(15)
        axes[0].barh(range(len(top_gain)), top_gain['importance_gain'], color='#2E86AB')
        axes[0].set_yticks(range(len(top_gain)))
        axes[0].set_yticklabels(top_gain['feature'])
        axes[0].invert_yaxis()
        axes[0].set_title('XGBoost Importance (Gain)', fontsize=12)
        
        top_perm = perm_importance_df.head(15)
        axes[1].barh(range(len(top_perm)), top_perm['importance_perm'], color='#A23B72')
        axes[1].set_yticks(range(len(top_perm)))
        axes[1].set_yticklabels(top_perm['feature'])
        axes[1].invert_yaxis()
        axes[1].set_title('Permutation Importance', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_plot_path = os.path.join(INPUT_FOLDER_MODEL_OUTPUT, f"{market_name}_shap_summary.png")
        plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to: {output_plot_path}")

    print("\nALL FEATURE IMPORTANCE ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()