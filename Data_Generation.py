import yfinance as yf
import pandas as pd
import os
import time

nifty50_tickers = [
    "ADANIENT.NS", "APOLLOHOSP.NS", "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS",
    "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HINDALCO.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDUNILVR.NS",
    "HEROMOTOCO.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JINDALSTEL.NS", "KOTAKBANK.NS",
    "LT.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "DRREDDY.NS",
    "RELIANCE.NS", "SBIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TCS.NS", "TECHM.NS",
    "TATASTEEL.NS", "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS"
]

eurostoxx50_tickers = [
    "ABI.BR", "ADS.DE", "AD.AS", "AIR.PA", "AI.PA", "ALV.DE", "ASML.AS", "AXA.PA", "SAN.MC", "BAS.DE",
    "BMW.DE", "BAYN.DE", "BBVA.MC", "BNP.PA", "CRH.L", "BN.PA", "DPW.DE", "DB1.DE", "DTE.DE", "ENEL.MI",
    "ENI.MI", "EL.PA", "FLTR.IR", "RMS.PA", "IBE.MC", "ITX.MC", "IFX.DE", "INGA.AS", "ISP.MI", "KER.PA",
    "OR.PA", "MC.PA", "MTX.DE", "MUV2.DE", "NOKIA.HE", "NDA-SE.ST", "RI.PA", "SAF.PA", "SGO.PA", "SAP.DE",
    "SU.PA", "ENR.DE", "STLAM.MI", "HO.PA", "GLE.PA", "VOW3.DE", "VNA.DE"
]

start_date = "2021-01-01"
end_date = "2025-09-30"
output_folder = "Dataset"
os.makedirs(output_folder, exist_ok=True)

def download_data(ticker_list, folder):
    for ticker in ticker_list:
        try:
            print(f"Downloading {ticker}...")
            safe_ticker_name = ticker.replace('.', '_')
            file_path = os.path.join(folder, f"{safe_ticker_name}_ohlcv.csv")

            data = yf.download(ticker, start=start_date, end=end_date)

            if not data.empty:
                data.to_csv(file_path)
            else:
                print(f"No data for {ticker}")
            time.sleep(0.5) 
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

def main():
    nifty_folder = os.path.join(output_folder, "Nifty50")
    euro_folder = os.path.join(output_folder, "EuroStoxx50")
    os.makedirs(nifty_folder, exist_ok=True)
    os.makedirs(euro_folder, exist_ok=True)

    print("DOWNLOADING NIFTY 50 DATA")
    download_data(nifty50_tickers, nifty_folder)
    
    print("\nDOWNLOADING EURO STOXX 50 DATA")
    download_data(eurostoxx50_tickers, euro_folder)
    
    print("\nDOWNLOAD COMPLETE!")

if __name__ == "__main__":
    main()