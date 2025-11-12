"""
src/utils/plotting.py

Visualization utilities for model diagnostics, predictions, and reports.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# 🎯 Basic plot setup
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 🧮 Training diagnostics
# ---------------------------------------------------------------------------

def plot_training_curve(train_losses, val_losses, save_path=None):
    """Plot training vs validation loss curve."""
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📊 Saved training curve → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 📈 Predictions vs Actual
# ---------------------------------------------------------------------------

def plot_predictions(y_true, y_pred, dates=None, save_path=None):
    """Plot predicted vs actual prices."""
    plt.figure()
    if dates is not None and len(dates) == len(y_true):
        plt.plot(dates, y_true, label="Actual", linewidth=2)
        plt.plot(dates, y_pred, label="Predicted", linestyle="--", linewidth=2)
    else:
        plt.plot(y_true, label="Actual")
        plt.plot(y_pred, label="Predicted", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Predicted vs Actual Price")
    plt.legend()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📈 Saved predicted vs actual plot → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 🔥 Feature correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, save_path=None):
    """Plot correlation heatmap for DataFrame."""
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
        print(f"🧠 Saved correlation heatmap → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 🧠 SHAP summary plot
# ---------------------------------------------------------------------------

def plot_shap_summary(shap_values, feature_names, save_path=None):
    """Generate SHAP summary bar plot."""
    import shap
    shap_values = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_names, "importance": shap_values})
    shap_df = shap_df.sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x="importance", y="feature", data=shap_df, palette="viridis")
    plt.title("Feature Importance (Mean |SHAP|)")
    plt.xlabel("Average absolute SHAP value")
    plt.ylabel("Feature")
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
        print(f"🧩 Saved SHAP summary plot → {save_path}")
    plt.close()
