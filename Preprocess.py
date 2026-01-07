import os
import pandas as pd
import numpy as np
import pandas_ta as ta

# --- Config ---
input_folder = "Dataset"
output_folder = "Processed_Data"
os.makedirs(output_folder, exist_ok=True)

future_holding_period = 5
profit_threshold = 0.05  
jfea_n_pct_change = 2
jfea_n_frac_change = 5

# --- Feature Functions ---
def percentage_change(series, N):
    pct_change = series.copy()
    pct_change.iloc[:N] = 0
    denom = series.shift(N).replace(0, np.nan)
    pct_change.iloc[N:] = (series.iloc[N:].values - denom.iloc[N:].values) / denom.iloc[N:].values
    pct_change.replace([np.inf, -np.inf], 0, inplace=True)
    pct_change.fillna(0, inplace=True)
    return pct_change

def fractional_change(series, N):
    frac_change = series.copy()
    frac_change.iloc[:N] = 0
    denom = series.shift(N).replace(0, np.nan)
    frac_change.iloc[N:] = series.iloc[N:].values / denom.iloc[N:].values - 1
    frac_change.replace([np.inf, -np.inf], 0, inplace=True)
    frac_change.fillna(0, inplace=True)
    return frac_change

# --- Label Generation ---
def create_labels(close_series):
    future_close = close_series.shift(-future_holding_period)
    forward_return = (future_close - close_series) / close_series
    labels = (forward_return > profit_threshold).astype(int)
    return labels

# --- Main Processing Function ---
def process_ticker_file(filepath, ticker_name):
    raw_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    raw_df.index.name = 'Date'

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')

    raw_df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    if raw_df.empty:
        return None

    df_feat = pd.DataFrame(index=raw_df.index)
    df_feat['Ticker'] = ticker_name

    # 1. JFEA Features
    df_feat['PctChange_Open'] = percentage_change(raw_df['Open'], N=jfea_n_pct_change)
    df_feat['PctChange_High'] = percentage_change(raw_df['High'], N=jfea_n_pct_change)
    df_feat['PctChange_Low'] = percentage_change(raw_df['Low'], N=jfea_n_pct_change)
    df_feat['PctChange_Close'] = percentage_change(raw_df['Close'], N=jfea_n_pct_change)
    df_feat['PctChange_Volume'] = percentage_change(raw_df['Volume'], N=jfea_n_pct_change)
    df_feat['FracChange_Open'] = fractional_change(raw_df['Open'], N=jfea_n_frac_change)
    df_feat['FracChange_Volume'] = fractional_change(raw_df['Volume'], N=jfea_n_frac_change)
    
    
    # 2. Technical Indicators
    df_feat['SMA_5'] = ta.sma(raw_df['Close'], length=5)
    df_feat['EMA_10'] = ta.ema(raw_df['Close'], length=10)
    df_feat['RSI_14'] = ta.rsi(raw_df['Close'], length=14)
    
    macd_df = ta.macd(raw_df['Close'])
    df_feat['MACD'] = macd_df.iloc[:, 0]
    df_feat['MACD_signal'] = macd_df.iloc[:, 1] 
    
    df_feat['ATR_14'] = ta.atr(raw_df['High'], raw_df['Low'], raw_df['Close'], length=14)
    
    bbands_df = ta.bbands(raw_df['Close'])
    df_feat['BBL_20'] = bbands_df.iloc[:, 0]
    df_feat['BBU_20'] = bbands_df.iloc[:, 2]

    stoch_df = ta.stoch(raw_df['High'], raw_df['Low'], raw_df['Close'])
    df_feat['STOCHk'] = stoch_df.iloc[:, 0]
    df_feat['STOCHd'] = stoch_df.iloc[:, 1]

    df_feat['Volume_ROC_10'] = ta.roc(raw_df['Volume'], length=10)
    df_feat['OBV'] = ta.obv(raw_df['Close'], raw_df['Volume'])

    # 3. Create Target Labels
    df_feat['Label'] = create_labels(raw_df['Close'])

    # 4. Clean up
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_feat.dropna(inplace=True)

    return df_feat

def main():
    market_folders = ["Nifty50", "EuroStoxx50"]

    for market in market_folders:
        print(f"\nPROCESSING {market.upper()}")
        market_input_folder = os.path.join(input_folder, market)
        all_ticker_data = []
        files = [f for f in os.listdir(market_input_folder) if f.endswith('_ohlcv.csv')]

        for file in files:
            filepath = os.path.join(market_input_folder, file)
            ticker_name = file.split('_ohlcv.csv')[0]
            print(f"Processing {ticker_name}...")

            processed_df = process_ticker_file(filepath, ticker_name)
            if processed_df is not None:
                all_ticker_data.append(processed_df)
                
        if not all_ticker_data:
            print(f"\nWARNING: No data processed for {market}.")
            continue

        combined_market_df = pd.concat(all_ticker_data)

        output_filename = f"{market.lower()}_processed_unscaled.csv"
        output_path = os.path.join(output_folder, output_filename)
        combined_market_df.to_csv(output_path)
        
        print(f"\nSaved {market} data to {output_path}")

if __name__ == "__main__":
    print("FEATURE ENGINEERING PIPELINE")
    main()
    print("\nPREPROCESSING COMPLETE!")