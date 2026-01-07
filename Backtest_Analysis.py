import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
INPUT_FOLDER_MODEL_OUTPUT = "Model_Output"
INPUT_FOLDER_RAW_DATA = "Dataset"
RISK_FREE_RATE = 0.04

MARKETS_TO_ANALYZE = {
    "nifty50": {
        "asset_folder": "Dataset/Nifty50",
        "benchmark_file": "nifty50_benchmark_index.csv",
        "benchmark_name": "Nifty 50"
    },
    "eurostoxx50": {
        "asset_folder": "Dataset/EuroStoxx50",
        "benchmark_file": "eurostoxx50_benchmark_index.csv",
        "benchmark_name": "EuroStoxx 50"
    }
}

# --- Custom metric functions ---
def sharpe_ratio(returns, risk_free_rate=0.04):
    excess = returns - risk_free_rate / 252
    if excess.std() == 0 or np.isnan(excess.std()): return np.nan
    return np.sqrt(252) * excess.mean() / excess.std()

def sortino_ratio(returns, risk_free_rate=0.04):
    excess = returns - risk_free_rate / 252
    downside_returns = excess[excess < 0]
    if len(downside_returns) == 0: return 0.0
    downside_std = np.sqrt(np.mean(downside_returns**2))
    if downside_std == 0 or np.isnan(downside_std): return np.nan
    annualized_mean = excess.mean() * 252
    annualized_downside_std = downside_std * np.sqrt(252)
    return annualized_mean / annualized_downside_std

def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

# --- Load all asset prices ---
def load_all_price_data(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('_ohlcv.csv')]
    price_data = []
    for file in files:
        ticker = file.split('_ohlcv.csv')[0]
        try:
            df = pd.read_csv(os.path.join(folder_path, file), index_col=0, parse_dates=False)
            df.index.name = 'Date'
            price_data.append(df[['Close']].rename(columns={'Close': ticker}))
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not price_data:
        raise ValueError("No data loaded. Check file paths.")
        
    prices_df = pd.concat(price_data, axis=1).dropna(how='all', axis=1)
    prices_df = prices_df.apply(pd.to_numeric, errors='coerce')
    prices_df.index = pd.to_datetime(prices_df.index, errors='coerce')
    prices_df = prices_df.dropna(how='all', axis=0)
    return prices_df

# --- Main Backtest Function ---
def main():
    
    for market_name, paths in MARKETS_TO_ANALYZE.items():
        
        print(f"\nRUNNING ANALYSIS FOR: {market_name.upper()}")

        # 1. Load Weights
        weights_path = os.path.join(INPUT_FOLDER_MODEL_OUTPUT, f"{market_name}_historical_weights.csv")
        try:
            weights_df = pd.read_csv(weights_path, index_col=0, parse_dates=True)
        except FileNotFoundError:
            print(f"ERROR: Weights file not found at {weights_path}. Run Black_Litterman.py first.")
            continue
        
        # 2. Load Asset Prices
        asset_prices_df = load_all_price_data(paths["asset_folder"])
        
        # 3. Load Benchmark
        benchmark_path = os.path.join(INPUT_FOLDER_RAW_DATA, paths["benchmark_file"])
        try:
            benchmark_prices_df = pd.read_csv(benchmark_path, index_col=0, parse_dates=False)
        except FileNotFoundError:
            print(f"ERROR: Benchmark file not found at {benchmark_path}. Run download_benchmark.py first.")
            continue
            
        benchmark_prices_df.index = pd.to_datetime(benchmark_prices_df.index, errors='coerce')
        benchmark_prices_df = benchmark_prices_df.dropna(how='all', axis=0)
        benchmark_prices_df['Close'] = pd.to_numeric(benchmark_prices_df['Close'], errors='coerce')
        benchmark_prices = benchmark_prices_df['Close']

        # 4. Prepare Data
        start_date = weights_df.index.min()
        end_date = weights_df.index.max()
        print(f"Backtest period: {start_date.date()} to {end_date.date()}")

        asset_prices_test = asset_prices_df[
            (asset_prices_df.index >= start_date) & (asset_prices_df.index <= end_date)
        ].ffill().bfill()
        asset_returns = asset_prices_test.pct_change().fillna(0)

        benchmark_prices_test = benchmark_prices[
            (benchmark_prices.index >= start_date) & (benchmark_prices.index <= end_date)
        ].ffill().bfill()
        benchmark_returns = benchmark_prices_test.pct_change().fillna(0)

        weights_shifted = weights_df.shift(1).fillna(0)
        aligned_weights, aligned_returns = weights_shifted.align(asset_returns, join='inner', axis=0)

        # 5. Calculate Returns
        print("Calculating portfolio returns...")
        portfolio_returns = (aligned_weights * aligned_returns[aligned_weights.columns]).sum(axis=1)
        
        # 6. Calculate Performance Metrics
        print("\n--- Strategy Performance ---")
        print(f"Sharpe Ratio: {sharpe_ratio(portfolio_returns):.2f}")
        print(f"Sortino Ratio: {sortino_ratio(portfolio_returns):.2f}")
        print(f"Max Drawdown: {max_drawdown(portfolio_returns) * 100:.2f}%")
        
        print(f"\n--- Benchmark ({paths['benchmark_name']}) ---")
        print(f"Sharpe Ratio: {sharpe_ratio(benchmark_returns):.2f}")
        print(f"Sortino Ratio: {sortino_ratio(benchmark_returns):.2f}")
        print(f"Max Drawdown: {max_drawdown(benchmark_returns) * 100:.2f}%")
        
        # 7. Generate Plot
        print("\nGenerating performance plot...")
        cum_portfolio = (1 + portfolio_returns).cumprod()
        cum_benchmark = (1 + benchmark_returns).cumprod()

        plt.figure(figsize=(12, 7))
        cum_portfolio.plot(label='Strategy')
        cum_benchmark.plot(label=paths['benchmark_name'])
        plt.title(f'Cumulative Returns: {market_name.upper()}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Growth (1 = 100%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(INPUT_FOLDER_MODEL_OUTPUT, f"{market_name}_performance.png")
        plt.savefig(save_path)
        print(f"Performance plot saved to {save_path}")
        
    
    print("\nALL BACKTEST ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()