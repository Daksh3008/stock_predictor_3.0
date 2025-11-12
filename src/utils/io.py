"""
src/utils/io.py

Utility functions for:
- Safe file and directory operations
- Config loading (YAML/JSON)
- Saving/loading DataFrames and experiment artifacts
- Getting latest files for model checkpointing

Author: Daksh Project - Deepak Nitrite LSTM Pipeline
"""

import os
import json
import yaml
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# 🧱 Directory & path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """Ensure a directory exists (creates recursively if not)."""
    os.makedirs(path, exist_ok=True)
    return path


def timestamp_str() -> str:
    """Return current timestamp as string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# ⚙️ Config I/O
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    """Load a YAML configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ YAML config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def save_json(obj: dict, path: str, indent: int = 2):
    """Save dictionary as formatted JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    print(f"💾 JSON saved → {path}")


def load_json(path: str) -> dict:
    """Load JSON file safely."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 📊 DataFrame I/O
# ---------------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, path: str, index: bool = True):
    """Save DataFrame to CSV (creates directories automatically)."""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index)
    print(f"✅ DataFrame saved → {path} ({len(df)} rows × {len(df.columns)} cols)")


def load_dataframe(path: str, parse_dates: bool = True) -> pd.DataFrame:
    """Load CSV into DataFrame with optional date parsing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=[0] if parse_dates else None, index_col=0)
    print(f"📥 Loaded DataFrame → {path} ({len(df)} rows × {len(df.columns)} cols)")
    return df


# ---------------------------------------------------------------------------
# 🧩 Utility helpers
# ---------------------------------------------------------------------------

def get_latest_file(folder: str, pattern: str = "") -> str:
    """Return most recently modified file in folder, optionally filtered by pattern."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"❌ Folder not found: {folder}")

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and pattern in f
    ]
    if not files:
        raise FileNotFoundError(f"⚠️ No files found in {folder} matching '{pattern}'")

    latest = max(files, key=os.path.getmtime)
    print(f"🕐 Latest file matching '{pattern}' → {latest}")
    return latest


def safe_read_csv(path: str) -> pd.DataFrame:
    """Read CSV with fallback encodings."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"⚠️ File not found: {path}")
        return pd.DataFrame()
    except UnicodeDecodeError:
        print(f"⚠️ Encoding issue, retrying ISO-8859-1 for {path}")
        return pd.read_csv(path, encoding="ISO-8859-1")


# ---------------------------------------------------------------------------
# 📘 Experiment logging
# ---------------------------------------------------------------------------

def log_experiment(metrics: dict, save_dir: str, filename: str = "experiment_log.json"):
    """Save experiment metrics with timestamp to JSON."""
    ensure_dir(save_dir)
    log_path = os.path.join(save_dir, filename)
    metrics["timestamp"] = timestamp_str()
    save_json(metrics, log_path)
    print(f"🧾 Experiment log saved → {log_path}")
    return log_path


def list_dir_files(folder: str, ext: str = None) -> list:
    """Return list of files in directory (optionally filter by extension)."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"❌ Folder not found: {folder}")
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and (ext is None or f.endswith(ext))
    ]
    return sorted(files)

# ---------------------------------------------------------------------------
# 🧮 Scaler persistence (StandardScaler or MinMaxScaler)
# ---------------------------------------------------------------------------

import joblib

def save_scaler(scaler, path: str):
    """Save sklearn scaler object using joblib."""
    ensure_dir(os.path.dirname(path))
    joblib.dump(scaler, path)
    print(f"🧩 Scaler saved → {path}")


def load_scaler(path: str):
    """Load previously saved sklearn scaler."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Scaler file not found: {path}")
    scaler = joblib.load(path)
    print(f"📥 Loaded scaler from {path}")
    return scaler
