"""
src/datasets/train_val_split.py

Helpers to create train/validation splits for time-series:
 - single_chronological_split: simple cutoff
 - rolling_origin_splits: yields (train_idx, val_idx) pairs for backtest
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator

def single_chronological_split(df: pd.DataFrame, train_size: float = 0.8):
    """
    Split df by chronological order.
    Returns df_train, df_val (views).
    """
    n = len(df)
    split = int(n * train_size)
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def rolling_origin_splits(n_samples: int, initial_train_size: int, horizon: int = 1,
                          step: int = 1, max_splits: int = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate indices for rolling-origin (expanding window) splits.

    Args:
      n_samples: total length of dataset
      initial_train_size: number of samples to start training with
      horizon: validation set size per fold (commonly pred_horizon or a small window)
      step: how many indices to advance train start per fold
      max_splits: maximum number of folds to produce (None = until end)

    Yields:
      (train_idx, val_idx) as numpy arrays of indices
    """
    start = initial_train_size
    splits = 0
    while start + horizon <= n_samples:
        train_idx = np.arange(0, start)
        val_idx = np.arange(start, start + horizon)
        yield train_idx, val_idx
        start += step
        splits += 1
        if max_splits is not None and splits >= max_splits:
            break
