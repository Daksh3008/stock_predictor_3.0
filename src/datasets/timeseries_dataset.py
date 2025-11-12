"""
src/datasets/timeseries_dataset.py

Robust PyTorch Dataset for multivariate sliding-window time-series forecasting.

Usage:
    ds = TimeSeriesDataset(df, feature_cols=..., target_col='close',
                           seq_len=60, pred_horizon=20, scaler=scaler)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seq_len: int = 60,
        pred_horizon: int = 1,
        scaler: Optional[object] = None,
    ):
        """
        Args:
            df: pandas DataFrame indexed by date (or integer). Can contain extra columns.
            feature_cols: list of column names to use as input features (order matters).
            target_col: name of the target column (single column, scalar output).
            seq_len: lookback window length (number of timesteps in X).
            pred_horizon: forecast horizon in timesteps (1 => predict t+1, 20 => predict t+20).
            scaler: optional fitted sklearn-like scaler with transform() method (applied per-window).
        """
        # Defensive copy
        self.df = df.copy()
        # normalize column names to strings (and preserve case as given)
        self.df.columns = [str(c) for c in self.df.columns]

        # validate requested columns exist
        missing = [c for c in feature_cols + [target_col] if c not in self.df.columns]
        if missing:
            raise KeyError(f"Missing columns in dataframe required by dataset: {missing}")

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = int(seq_len)
        self.pred_horizon = int(pred_horizon)
        self.scaler = scaler

        # extract numpy arrays for speed
        self._features = self.df[self.feature_cols].values.astype(float)
        self._target = self.df[self.target_col].values.astype(float)

        # compute valid window start indices so we won't go out of bounds
        # we want windows where: start ... start+seq_len-1 are available, and
        # target index = start + seq_len - 1 + pred_horizon is available.
        N = len(self._features)
        min_required = self.seq_len + self.pred_horizon  # minimal rows needed to create one sample
        if N < min_required:
            raise ValueError(
                f"Not enough rows to build a single sample: N={N}, required={min_required} "
                f"(seq_len={self.seq_len}, pred_horizon={self.pred_horizon})"
            )

        # valid starts: 0 .. N - (seq_len + pred_horizon)
        last_start = N - (self.seq_len + self.pred_horizon)
        # inclusive range: 0 .. last_start
        self.starts = np.arange(0, last_start + 1, dtype=int)

        # quick sanity info
        print(f"TimeSeriesDataset: N={N}, seq_len={self.seq_len}, pred_horizon={self.pred_horizon}, samples={len(self.starts)}, features={len(self.feature_cols)}")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        """
        Returns:
            X: np.float32 array shape (seq_len, n_features)
            y: np.float32 scalar (the target value at t = start + seq_len -1 + pred_horizon)
        """
        start = int(self.starts[idx])
        end = start + self.seq_len  # exclusive
        target_idx = end + self.pred_horizon - 1

        X = self._features[start:end]  # shape (seq_len, n_features)
        y = self._target[target_idx]   # scalar

        # apply scaler if provided (scaler expects 2D array)
        if self.scaler is not None:
            # scaler.transform expects shape (n_rows, n_features) — we transform the sequence as rows
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                # fallback: try scaling per-feature manually (avoid crashing)
                # if scaler has mean_ and scale_ attributes, apply simple transform
                if hasattr(self.scaler, "mean_") and hasattr(self.scaler, "scale_"):
                    X = (X - self.scaler.mean_) / (self.scaler.scale_ + 1e-12)
                else:
                    # if transform fails and scaler is custom, just pass-through
                    pass

        return X.astype(np.float32), np.array(y, dtype=np.float32)
