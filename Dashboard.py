import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import warnings

# --- CONFIG ---
INPUT_FOLDER_MODEL_OUTPUT = "Model_Output"
INPUT_FOLDER_RAW_DATA = "Dataset"
RISK_FREE_RATE = 0.04

st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")

# --- NEW FUNCTION: Get Top 10 Combined ---
@st.cache_data
def get_top_combined_stocks():
    """Loads both weight files and returns the combined top 10 stocks."""
    all_avg_weights = pd.Series(dtype=float)
    
    # Load and process Nifty 50
    try:
        nifty_weights_path = os.path.join(INPUT_FOLDER_MODEL_OUTPUT, "nifty50_historical_weights.csv")
        nifty_df = pd.read_csv(nifty_weights_path, index_col=0, parse_dates=True)
        # Calculate mean weight for each stock
        nifty_avg = nifty_df.mean()
        all_avg_weights = pd.concat([all_avg_weights, nifty_avg])
    except FileNotFoundError:
        st.sidebar.warning("Nifty 50 weights file not found. Skipping.")
    except Exception as e:
        st.sidebar.error(f"Error loading Nifty 50 weights: {e}")

    # Load and process EuroStoxx 50
    try:
        euro_weights_path = os.path.join(INPUT_FOLDER_MODEL_OUTPUT, "eurostoxx50_historical_weights.csv")
        euro_df = pd.read_csv(euro_weights_path, index_col=0, parse_dates=True)
        # Calculate mean weight for each stock
        euro_avg = euro_df.mean()
        all_avg_weights = pd.concat([all_avg_weights, euro_avg])
    except FileNotFoundError:
        st.sidebar.warning("EuroStoxx 50 weights file not found. Skipping.")
    except Exception as e:
        st.sidebar.error(f"Error loading EuroStoxx 50 weights: {e}")
        
    # Get top 10
    top_10 = all_avg_weights.nlargest(10)
    
    if top_10.empty:
        return pd.DataFrame(columns=['Stock Ticker', 'Average Weight'])

    top_10_df = top_10.reset_index()
    top_10_df.columns = ['Stock Ticker', 'Average Weight']
    top_10_df.index = top_10_df.index + 1 # Start index at 1
    
    return top_10_df

# --- SIDEBAR FOR MARKET SELECTION ---
market_map = {
    "Nifty 50": "nifty50",
    "EuroStoxx 50": "eurostoxx50"
}
market_display_name = st.sidebar.selectbox("Select Market for Detailed View", ["Nifty 50", "EuroStoxx 50"])
MARKET_NAME = market_map[market_display_name]
INPUT_FOLDER_RAW_ASSETS = f"Dataset/{'Nifty50' if MARKET_NAME == 'nifty50' else 'EuroStoxx50'}"
BENCHMARK_NAME = "Nifty 50" if MARKET_NAME == "nifty50" else "EuroStoxx 50"

st.title("AI-Driven Portfolio Optimization Dashboard")
st.write("This dashboard showcases the strategy performance and model explainability.")

# --- NEW SECTION: TOP 10 COMBINED STOCKS ---
st.header("Top 10 Stocks (Combined Markets)")
st.write("These are the 10 stocks with the highest average portfolio weight across the entire backtest, combining both Nifty 50 and EuroStoxx 50.")
top_10_data = get_top_combined_stocks()

# Format the weight as a percentage string
if not top_10_data.empty:
    top_10_data['Average Weight'] = (top_10_data['Average Weight'] * 100).map('{:,.2f}%'.format)
    st.dataframe(top_10_data, use_container_width=True)
else:
    st.warning("No weight files found. Please run `Black_Litterman.py` for both markets.")

st.divider()

# --- OLD SECTIONS: DETAILED MARKET VIEW ---
st.header(f"Detailed Analysis: {market_display_name}")

# --- Helper Functions ---
@st.cache_data
def sharpe_ratio(returns, risk_free_rate=0.04):
    excess = returns - risk_free_rate / 252
    if excess.std() == 0 or np.isnan(excess.std()): return np.nan
    return np.sqrt(252) * excess.mean() / excess.std()

@st.cache_data
def sortino_ratio(returns, risk_free_rate=0.04):
    excess = returns - risk_free_rate / 252
    downside_returns = excess[excess < 0]
    if len(downside_returns) == 0: return 0.0
    downside_std = np.sqrt(np.mean(downside_returns**2))
    if downside_std == 0 or np.isnan(downside_std): return np.nan
    annualized_mean = excess.mean() * 252
    annualized_downside_std = downside_std * np.sqrt(252)
    return annualized_mean / annualized_downside_std

@st.cache_data
def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

@st.cache_data
def load_all_price_data(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('_ohlcv.csv')]
    price_data = []
    for file in files:
        ticker = file.split('_ohlcv.csv')[0]
        try:
            df = pd.read_csv(os.path.join(folder_path, file), index_col=0, parse_dates=False)
            df.index.name = 'Date'
            price_data.append(df[['Close']].rename(columns={'Close': ticker}))
        except Exception:
            pass
    prices_df = pd.concat(price_data, axis=1).dropna(how='all', axis=1)
    prices_df = prices_df.apply(pd.to_numeric, errors='coerce')
    prices_df.index = pd.to_datetime(prices_df.index, errors='coerce')
    prices_df = prices_df.dropna(how='all', axis=0)
    return prices_df

@st.cache_data
def load_all_data(market_name, asset_folder_path):
    """Loads all data for the selected market."""
    weights_path = os.path.join(INPUT_FOLDER_MODEL_OUTPUT, f"{market_name}_historical_weights.csv")
    if not os.path.exists(weights_path):
        return None, None, None
    weights_df = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    
    asset_prices_df = load_all_price_data(asset_folder_path)
    
    benchmark_path = os.path.join(INPUT_FOLDER_RAW_DATA, f"{market_name}_benchmark_index.csv")
    if not os.path.exists(benchmark_path):
        return weights_df, asset_prices_df, None
        
    benchmark_prices_df = pd.read_csv(benchmark_path, index_col=0, parse_dates=False)
    benchmark_prices_df.index = pd.to_datetime(benchmark_prices_df.index, errors='coerce')
    benchmark_prices_df.dropna(how='all', axis=0, inplace=True)
    benchmark_prices_df['Close'] = pd.to_numeric(benchmark_prices_df['Close'], errors='coerce')
    benchmark_prices = benchmark_prices_df['Close']
    
    return weights_df, asset_prices_df, benchmark_prices

# --- 1. Load Data for Selected Market ---
weights_df, asset_prices_df, benchmark_prices = load_all_data(MARKET_NAME, INPUT_FOLDER_RAW_ASSETS)

# --- 2. Build the Dashboard ---
if weights_df is None:
    st.error(f"Data for {market_display_name} not found. Please run the full pipeline (Model_Train.py, Black_Litterman.py, etc.).")
else:
    # Calculate Returns
    start_date = weights_df.index.min()
    end_date = weights_df.index.max()
    
    asset_prices_test = asset_prices_df[(asset_prices_df.index >= start_date) & (asset_prices_df.index <= end_date)].ffill().bfill()
    asset_returns = asset_prices_test.pct_change().fillna(0)
    
    benchmark_prices_test = benchmark_prices[(benchmark_prices.index >= start_date) & (benchmark_prices.index <= end_date)].ffill().bfill()
    benchmark_returns = benchmark_prices_test.pct_change().fillna(0)
    
    weights_shifted = weights_df.shift(1).fillna(0)
    aligned_weights, aligned_returns = weights_shifted.align(asset_returns, join='inner', axis=0)
    
    portfolio_returns = (aligned_weights * aligned_returns[aligned_weights.columns]).sum(axis=1)

    # == Section 1: Performance Metrics ==
    st.subheader("1. Final Performance Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("XGBoost-BL Strategy")
        st.metric("Sharpe Ratio", f"{sharpe_ratio(portfolio_returns):.2f}")
        st.metric("Sortino Ratio", f"{sortino_ratio(portfolio_returns):.2f}")
        st.metric("Max Drawdown", f"{max_drawdown(portfolio_returns) * 100:.2f}%")
        
    with col2:
        st.subheader(f"{BENCHMARK_NAME} Benchmark")
        st.metric("Sharpe Ratio", f"{sharpe_ratio(benchmark_returns):.2f}")
        st.metric("Sortino Ratio", f"{sortino_ratio(benchmark_returns):.2f}")
        st.metric("Max Drawdown", f"{max_drawdown(benchmark_returns) * 100:.2f}%")

    st.divider()

    # == Section 2: Cumulative Returns Plot ==
    st.subheader("2. Cumulative Returns (Strategy vs. Benchmark)")
    plot_image_path = os.path.join(INPUT_FOLDER_MODEL_OUTPUT, f"{MARKET_NAME}_performance.png")
    if os.path.exists(plot_image_path):
        plot_image = Image.open(plot_image_path)
        st.image(plot_image, caption=f"Cumulative returns for {market_display_name}.", use_container_width=True) 
    else:
        st.error(f"Performance plot not found. Please run Backtest_Analysis.py")

    st.divider()

    # == Section 3: Model Explainability ==
    st.subheader("3. Model Explainability (Feature Importance)")
    shap_image_path = os.path.join(INPUT_FOLDER_MODEL_OUTPUT, f"{MARKET_NAME}_shap_summary.png")
    if os.path.exists(shap_image_path):
        shap_image = Image.open(shap_image_path)
        st.image(shap_image, caption=f"Feature Importance for {market_display_name} model.", use_container_width=True)
    else:
        st.error(f"Feature importance plot not found. Please run generate_shap.py")

    st.divider()

    # == Section 4: Historical Portfolio Allocations ==
    st.subheader("4. Historical Portfolio Allocations")
    st.write(f"Top 15 most-held assets in the {market_display_name} portfolio.")
    
    top_assets = weights_df.mean().nlargest(15).index
    weights_to_plot = weights_df[top_assets]
    
    st.area_chart(weights_to_plot)
    st.subheader("Raw Weights Data (Sample)")
    st.dataframe(weights_df.head())
