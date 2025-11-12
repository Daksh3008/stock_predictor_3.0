"""Fetch CPI and RBI / repo rate. This module contains a couple of helper placeholders.
You can wire it to an API (TradingEconomics, FRED, RBI CSV) by adding credentials.
"""
import os
import pandas as pd
from datetime import datetime

RAW_DIR = os.path.join("..","..","data","raw")
os.makedirs(RAW_DIR, exist_ok=True)


def load_local_cpi(path: str = None):
    """If you have a local CPI CSV, load it. Otherwise the function returns an empty DF
    with a monthly datetime index for future merging.
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        return df
    # placeholder empty DF
    print("No local CPI provided. Please add CPI dataset or wire TradingEconomics API.")
    return pd.DataFrame()


# TODO: implement TradingEconomics/FRED fetchers if you want live pulls with API keys.