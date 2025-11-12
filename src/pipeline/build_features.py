"""
src/pipeline/build_features.py

Fetch raw data (stock, macro, news, corporate events),
compute technical indicators, merge everything, and save the processed feature matrix.

Usage:
    python -m src.pipeline.build_features
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from src.data_fetchers.corporate_events import (fetch_bse_announcements, fetch_nse_announcements, aggregate_daily_events) 
from ta import add_all_ta_features
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)


def fetch_stock(symbol="DEEPAKNTR.NS", start="2000-01-01", end=None):
    print("📥 Fetching stock data...")
    end = end or datetime.today().strftime("%Y-%m-%d")
    df = yf.download(symbol, start=start, end=end, progress=False)
    df.dropna(inplace=True)
    df.to_csv(os.path.join(RAW_DIR, "stock_deepak.csv"))
    print(f"✅ Saved stock data: {len(df)} rows")
    return df


def fetch_macro():
    print("🌍 Fetching macro data (Brent, USD/INR)...")

    def flatten_cols(df):
        """Flatten MultiIndex to single-level columns."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        else:
            df.columns = [str(c) for c in df.columns]
        return df

    # Fetch Brent
    brent = yf.download("BZ=F", start="2000-01-01", progress=False)
    brent = flatten_cols(brent)
    if "Close" not in brent.columns:
        brent.rename(columns={brent.columns[0]: "Close"}, inplace=True)
    brent = brent[["Close"]].rename(columns={"Close": "brent_close"})

    # Fetch USD/INR
    fx = yf.download("USDINR=X", start="2000-01-01", progress=False)
    fx = flatten_cols(fx)
    if "Close" not in fx.columns:
        fx.rename(columns={fx.columns[0]: "Close"}, inplace=True)
    fx = fx[["Close"]].rename(columns={"Close": "usd_inr"})

    macro = brent.join(fx, how="outer").ffill().bfill()
    macro.to_csv(os.path.join(RAW_DIR, "macro.csv"))
    print(f"✅ Saved macro data: {len(macro)} rows, columns={list(macro.columns)}")
    return macro



def compute_indicators(df):
    """
    Robust indicator computation (case-insensitive + self-healing).
    """
    print("🧮 Computing technical indicators (final robust mode)...")

    
    # --- Standardize column names ---
    
    # flatten multiIndex columns if needed ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).capitalize() for c in df.columns]
    else:
        df.columns = [str(c).strip().capitalize() for c in df.columns]

    # normalize common variants
    rename_map = {
        "Adj close": "Close",
        "Adjclose": "Close",
        "Adj_close": "Close",
    }  
    
    df.rename(columns=rename_map, inplace=True)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            #try lowercase fallback
            lower = col.lower()
            if lower in df.columns:
                df[col] = df[lower]
            else:
                raise ValueError(
                    f"❌ Required column '{col}' missing in DataFrame"
                    f"Columns found: {list(df.columns)}"
                )

        #ensure column is 1D
        if hasattr(df[col], "values") and getattr(df[col].values, "ndim", 1) > 1:
            df[col] = df[col].values.squeeze()
        
    #compute log returns 
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["pct_change"] = df["Close"].pct_change()


    # --- Technical indicators ---
    try:
        import ta

        # RSI
        df["ind_rsi_14"] = ta.momentum.rsi(df["Close"], window=14, fillna=True)

        # Moving averages
        for win in [10, 50, 200]:
            df[f"ind_ma{win}"] = df["Close"].rolling(window=win, min_periods=1).mean()
        df["ind_ema20"] = df["Close"].ewm(span=20, adjust=False).mean()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
        df["ind_bb_h"] = bb.bollinger_hband()
        df["ind_bb_l"] = bb.bollinger_lband()

        # Momentum
        df["ind_mom_10"] = ta.momentum.roc(df["Close"], window=10, fillna=True)

        # MACD
        macd = ta.trend.MACD(df["Close"])
        df["ind_macd"] = macd.macd()
        df["ind_macd_sig"] = macd.macd_signal()

    except Exception as e:
        print("⚠️ TA-lib warning:", e)
        # safe fallback
        df["ind_rsi_14"] = _rsi_fallback(df["Close"], 14)
        df["ind_ma10"] = df["Close"].rolling(10, min_periods=1).mean()
        df["ind_ma50"] = df["Close"].rolling(50, min_periods=1).mean()
        df["ind_ma200"] = df["Close"].rolling(200, min_periods=1).mean()
        df["ind_ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        ma20 = df["Close"].rolling(20, min_periods=1).mean()
        std20 = df["Close"].rolling(20, min_periods=1).std()
        df["ind_bb_h"] = ma20 + 2 * std20
        df["ind_bb_l"] = ma20 - 2 * std20
        df["ind_mom_10"] = df["Close"].pct_change(10)
        fast = df["Close"].ewm(span=12, adjust=False).mean()
        slow = df["Close"].ewm(span=26, adjust=False).mean()
        df["ind_macd"] = fast - slow
        df["ind_macd_sig"] = df["ind_macd"].ewm(span=9, adjust=False).mean()

    # --- Final cleanup ---
    valid_cols = [c for c in ["Close", "log_return"] if c in df.columns]
    if not valid_cols:
        raise RuntimeError("❌ Neither 'Close' nor 'log_return' found after processing!")

    df.dropna(subset=valid_cols, inplace=True)
    print(f"✅ Indicators computed. Final columns: {len(df.columns)} features, {len(df)} rows.")
    return df

#------------------------------------------------------------------------------
def _rsi_fallback(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)



def fetch_news_sentiment(symbol="Deepak Nitrite"):
    print("🗞️ Simulating news sentiment fetch (placeholder)...")
    # Placeholder: normally integrate Yahoo/GDELT API
    analyzer = SentimentIntensityAnalyzer()
    # For now generate neutral baseline sentiment
    dates = pd.date_range("2020-01-01", datetime.today(), freq="B")
    sentiment = pd.DataFrame({
        "date": dates,
        "news_compound": np.random.uniform(-0.1, 0.1, len(dates))
    }).set_index("date")
    sentiment.to_csv(os.path.join(RAW_DIR, "news_sentiment.csv"))
    print("✅ Saved placeholder news sentiment.")
    return sentiment


def merge_all():
    print("🔗 Merging all datasets...")
    stock = fetch_stock()
    stock = compute_indicators(stock)

    macro = fetch_macro()
    events_bse = fetch_bse_announcements()
    events_nse = fetch_nse_announcements()
    events_daily = aggregate_daily_events()

    # --- Handle case where no events were fetched ---
    if events_daily is None or len(events_daily) == 0:
        print("⚠️ No corporate events available; continuing with empty event features.")
        events_daily = pd.DataFrame(index=pd.to_datetime([]))

    news = fetch_news_sentiment()

    # Align by date index
    stock.index = pd.to_datetime(stock.index)
    macro.index = pd.to_datetime(macro.index)
    events_daily.index = pd.to_datetime(events_daily.index)
    news.index = pd.to_datetime(news.index)

    df = stock.join(macro, how="left").join(events_daily, how="left").join(news, how="left")
    df = df.ffill().bfill()

    out_path = os.path.join(PROC_DIR, "feature_matrix.csv")
    df.to_csv(out_path)
    print(f"✅ Final feature matrix saved: {len(df)} rows → {out_path}")


def main():
    print("=" * 80)
    print("🏗️  Building Deepak Nitrite feature matrix...")
    print("=" * 80)
    merge_all()
    print("✅ All features built successfully!")


if __name__ == "__main__":
    main()
