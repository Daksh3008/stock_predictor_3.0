"""Cleaner and merger: read raw CSVs, align to business days, forward-fill missing macro values,
compute log returns, and save merged intermediate.
"""
import os
import pandas as pd
import numpy as np

ROOT = os.path.join("..","..")
RAW_DIR = os.path.join(ROOT, "data", "raw")
INTER_DIR = os.path.join(ROOT, "data", "intermediates")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
for p in (INTER_DIR, PROCESSED_DIR):
    os.makedirs(p, exist_ok=True)


def load_csv_safe(path: str):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def build_merged_dataframe():
    stock = load_csv_safe(os.path.join(RAW_DIR, "deepak_raw.csv"))
    brent = load_csv_safe(os.path.join(RAW_DIR, "brent_raw.csv"))
    fx = load_csv_safe(os.path.join(RAW_DIR, "usd_inr_raw.csv"))
    news = load_csv_safe(os.path.join(RAW_DIR, "news_headlines_yahoo.csv"))

    if stock.empty:
        raise RuntimeError("Stock raw data missing. Run data_fetchers.stock_yfinance.fetch_deepak() first.")

    # pick columns of interest
    s = stock[['Open','High','Low','Close','Adj Close','Volume']].copy()
    s.columns = [c.lower().replace(' ', '_') for c in s.columns]

    # resample/align brent and fx to stock index (business days). Use Close price for macro
    macro_df = pd.DataFrame(index=s.index)
    if not brent.empty:
        macro_df['brent_close'] = brent['Close'].reindex(s.index).fillna(method='ffill')
    if not fx.empty:
        macro_df['usd_inr'] = fx['Close'].reindex(s.index).fillna(method='ffill')

    # CPI & repo: placeholder (monthly). We'll merge if available and forward-fill

    # Aggregate news to daily compound score
    if not news.empty:
        news['datetime_utc'] = pd.to_datetime(news['datetime_utc'])
        news['date'] = news['datetime_utc'].dt.date
        daily_news = news.groupby('date')['compound'].mean()
        daily_news.index = pd.to_datetime(daily_news.index)
        macro_df['news_compound'] = daily_news.reindex(s.index).fillna(0)
    else:
        macro_df['news_compound'] = 0.0

    # combine
    df = pd.concat([s, macro_df], axis=1)

    # compute returns
    df['log_return'] = np.log(df['close']).diff()
    df['pct_change'] = df['close'].pct_change()

    # fill missing macro via forward fill
    df.ffill(inplace=True)

    # drop initial NaNs from returns
    df.dropna(subset=['log_return'], inplace=True)

    out_path = os.path.join(INTER_DIR, 'merged_intermediate.csv')
    df.to_csv(out_path)
    print(f"Saved merged intermediate to {out_path} rows={len(df)}")
    return df