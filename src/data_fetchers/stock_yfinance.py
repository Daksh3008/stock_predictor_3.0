"""Fetch historical OHLCV for DEEPAKNTR (NSE ticker) using yfinance.
Saves CSV to data/raw/deepak_stock.csv
"""
import os
import yfinance as yf
import pandas as pd
from datetime import datetime

RAW_DIR = os.path.join("..","..","data","raw")
os.makedirs(RAW_DIR, exist_ok=True)


def fetch_deepak(start_date: str = "2000-01-01", end_date: str = None):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    # Yahoo uses NSE tickers like 'DEEPAKNTR.NS' for Indian stocks
    ticker = "DEEPAKNTR.NS"
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")
    df.index = pd.to_datetime(df.index)
    # ensure columns: Open, High, Low, Close, Adj Close, Volume
    out_path = os.path.join(RAW_DIR, "deepak_raw.csv")
    df.to_csv(out_path)
    print(f"Saved stock raw to {out_path} rows={len(df)}")
    return df


if __name__ == "__main__":
    fetch_deepak()