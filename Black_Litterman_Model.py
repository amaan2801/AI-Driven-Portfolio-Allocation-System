import xgboost as xgb
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from pypfopt import BlackLittermanModel, EfficientFrontier, risk_models, expected_returns, market_implied_risk_aversion
from scipy.stats import norm
import warnings

# --- CONFIG ---
INPUT_FOLDER_PROCESSED = "Processed_Data" 
INPUT_FOLDER_MODEL_DATA = "Model_Input"
INPUT_FOLDER_MODEL_OUTPUT = "Model_Output"

MARKETS_TO_RUN = {
    "nifty50": "Dataset/Nifty50",
    "eurostoxx50": "Dataset/EuroStoxx50"
}
SPLIT_DATE = '2022-05-01'

# --- 1. DATA LOADING ---
def load_all_price_data(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('_ohlcv.csv')]
    price_data = []
    
    for file in all_files:
        ticker = file.split('_ohlcv.csv')[0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                df = pd.read_csv(os.path.join(folder_path, file), index_col=0, parse_dates=True)
            df.index.name = 'Date'
            price_data.append(df[['Close']].rename(columns={'Close': ticker}))
        except Exception as e:
            print(f"Warning: Could not load {file}. Error: {e}")
            
    if not price_data:
        raise ValueError(f"No price data loaded from {folder_path}.")
        
    prices_df = pd.concat(price_data, axis=1)
    prices_df = prices_df.dropna(how='all', axis=1)
    prices_df = prices_df.apply(pd.to_numeric, errors='coerce')
    return prices_df

def get_scaled_features(market_name):
    file_path = os.path.join(INPUT_FOLDER_PROCESSED, f"{market_name}_processed_unscaled.csv")
    processed_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    
    features_to_drop = ['Label', 'Ticker']
    features = [col for col in processed_df.columns if col not in features_to_drop]
    
    train_df = processed_df[processed_df.index < SPLIT_DATE]
    test_df = processed_df[processed_df.index >= SPLIT_DATE]

    scaler = MinMaxScaler()
    scaler.fit(train_df[features]) 
    scaled_test_features = scaler.transform(test_df[features])
    
    scaled_test_df = pd.DataFrame(
        scaled_test_features, 
        columns=features, 
        index=test_df.index
    )
    scaled_test_df['Ticker'] = test_df['Ticker']
    return scaled_test_df

# --- 2. OPTIMIZATION FUNCTION ---
def run_optimization_for_day(day, model, S_full, pi_prior, test_df, tau, delta):
    
    features_today_all = test_df.loc[day]
    common_tickers = list(set(S_full.columns) & set(features_today_all['Ticker']))
    
    if len(common_tickers) == 0:
        return pd.Series(name=day) # Return empty Series for this day
    
    S_day = S_full.loc[common_tickers, common_tickers]
    pi_day = pi_prior.loc[common_tickers] 
    features_today = features_today_all.set_index('Ticker').loc[common_tickers]
    features_today_aligned = features_today.loc[S_day.columns]
    
    probs = model.predict_proba(features_today_aligned.values)[:, 1]
    
    P = np.identity(len(common_tickers))
    k = 0.1 
    vol = np.sqrt(np.diag(S_day)) 
    prob_z_scores = norm.ppf(np.clip(probs, 1e-3, 0.999)) - norm.ppf(0.5) 
    Q = pi_day + k * prob_z_scores * vol
    
    confidence = 1 - 4 * (probs - 0.5)**2
    uncertainty_factor = confidence  
    omega_diag = tau * (vol**2) * uncertainty_factor
    Omega = np.diag(omega_diag)

    bl = BlackLittermanModel(
        S_day, pi=pi_day, Q=Q.values, P=P, Omega=Omega, tau=tau, risk_aversion=delta
    )
    posterior_returns = bl.bl_returns()
    
    ef = EfficientFrontier(posterior_returns, S_day)
    ef.add_constraint(lambda w: w.sum() == 1) 
    ef.add_constraint(lambda w: w >= 0)      
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    return pd.Series(cleaned_weights, name=day)

# --- 3. MAIN BACKTESTING FUNCTION ---
def main():
    for market_name, raw_folder_path in MARKETS_TO_RUN.items():
        
        print(f"\nRUNNING BACKTEST FOR: {market_name.upper()}")
        
        # Load model
        print("Loading trained model...")
        model_path = os.path.join(INPUT_FOLDER_MODEL_OUTPUT, f"{market_name}_best_model.joblib")
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"ERROR: Model file not found at {model_path}. Run Model_Train.py first.")
            continue
        
        # Load price data
        print("Loading price and feature data...")
        prices_df = load_all_price_data(raw_folder_path)
        scaled_test_df = get_scaled_features(market_name)

        # Calculate priors
        print("Calculating market priors...")
        prices_train = prices_df[prices_df.index < SPLIT_DATE].ffill().dropna(axis=1, how='any')
        S_full = risk_models.CovarianceShrinkage(prices_train).ledoit_wolf()
        
        market_returns = prices_train.pct_change().mean(axis=1).dropna()
        delta = market_implied_risk_aversion(market_returns)
        if delta <= 0:
            print(f"Calculated delta is {delta:.2f}. Using fallback 2.5")
            delta = 2.5
        
        n_assets = len(S_full.columns)
        w_mkt = np.full(n_assets, 1.0 / n_assets) 
        pi_prior_full = delta * S_full.dot(w_mkt)
        pi_prior_full = pd.Series(pi_prior_full, index=S_full.columns) 

        tau = 0.05 
        
        # Run backtest loop
        print(f"Running backtest loop for {market_name.upper()}...")
        optimization_days = scaled_test_df.index.unique().sort_values()
        all_weights_list = []
        
        for day in optimization_days:
            daily_weights = run_optimization_for_day(
                day, model, S_full, pi_prior_full, scaled_test_df, tau, delta
            )
            all_weights_list.append(daily_weights)
        
        print("Backtest complete.")

        # Consolidate and save
        all_weights_df = pd.concat(all_weights_list, axis=1).T
        all_weights_df.fillna(0, inplace=True)
        
        output_path = os.path.join(INPUT_FOLDER_MODEL_OUTPUT, f"{market_name}_historical_weights.csv")
        all_weights_df.to_csv(output_path)
        
        print(f"Historical weights for {market_name} saved to: {output_path}")

if __name__ == "__main__":
    main()