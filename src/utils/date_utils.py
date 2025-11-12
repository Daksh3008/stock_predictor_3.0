"""
src/utils/date_utils.py

Date and time helper utilities for financial time series alignment.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# 🧮 Basic date helpers
# ---------------------------------------------------------------------------

def today_str(fmt: str = "%Y-%m-%d") -> str:
    """Return today's date as a string."""
    return datetime.today().strftime(fmt)


def n_days_ago(n: int, fmt: str = "%Y-%m-%d") -> str:
    """Return date string for n days ago."""
    return (datetime.today() - timedelta(days=n)).strftime(fmt)


def date_range(start: str, end: str, freq: str = "B") -> pd.DatetimeIndex:
    """Return a pandas date range (default: business days)."""
    return pd.date_range(start=start, end=end, freq=freq)


def is_business_day(date: datetime) -> bool:
    """Return True if a given date is a business day (Mon–Fri)."""
    return bool(len(pd.bdate_range(date, date)))


# ---------------------------------------------------------------------------
# 🧩 Alignment helpers
# ---------------------------------------------------------------------------

def align_to_business_days(df: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
    """Forward-fill and align DataFrame to continuous business-day index."""
    if start is None:
        start = df.index.min()
    if end is None:
        end = df.index.max()

    all_days = pd.bdate_range(start=start, end=end)
    df_aligned = df.reindex(all_days).ffill()
    df_aligned.index.name = "Date"
    return df_aligned


def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure time series has no missing dates (fill with NaN and forward-fill)."""
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return df.reindex(full_idx).ffill()


def shift_forward(df: pd.DataFrame, col: str, days: int = 1) -> pd.Series:
    """Return a shifted forward series (useful for horizon-aligned targets)."""
    return df[col].shift(-days)


# ---------------------------------------------------------------------------
# 🕹️ Rolling window helpers
# ---------------------------------------------------------------------------

def rolling_corr(series1: pd.Series, series2: pd.Series, window: int = 30) -> pd.Series:
    """Compute rolling correlation between two series."""
    return series1.rolling(window).corr(series2)


def rolling_mean(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling mean (simple moving average)."""
    return series.rolling(window).mean()


def rolling_std(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation."""
    return series.rolling(window).std()
