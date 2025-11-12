"""Fetch Brent crude futures via yfinance ticker 'BZ=F'.
"""
import os
import yfinance as yf
import pandas as pd
from datetime import datetime

RAW_DIR = os.path.join("..","..","data","raw")
os.makedirs(RAW_DIR, exist_ok=True)


def fetch_brent(start_date: str = "2000-01-01", end_date: str = None):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    ticker = "BZ=F"
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df.index = pd.to_datetime(df.index)
    out_path = os.path.join(RAW_DIR, "brent_raw.csv")
    df.to_csv(out_path)
    print(f"Saved brent raw to {out_path} rows={len(df)}")
    return df


if __name__ == "__main__":
    fetch_brent()