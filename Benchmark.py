import yfinance as yf
import os

# Define benchmarks
BENCHMARKS = {
    "nifty50": "^NSEI",
    "eurostoxx50": "^STOXX50E"
}

START_DATE = "2021-01-01"
END_DATE = "2025-09-30"
OUTPUT_FOLDER = "Dataset"

def download_benchmark(market_name, ticker):
    print(f"DOWNLOADING BENCHMARK: {market_name.upper()} ({ticker})")
    
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE)
        if data.empty:
            print(f"ERROR: No data downloaded for {ticker}.")
            return

        output_filename = f"{market_name}_benchmark_index.csv"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        data.to_csv(output_path)
        print(f"Successfully saved benchmark data to {output_path}")

    except Exception as e:
        print(f"ERROR: An error occurred: {e}")

if __name__ == "__main__":
    for market, ticker in BENCHMARKS.items():
        download_benchmark(market, ticker)
    print("\nALL BENCHMARK DOWNLOADS COMPLETE!")