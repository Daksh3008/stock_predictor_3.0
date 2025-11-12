"""
src/reporting/report_generator.py

Generates a detailed executive-style report after model training or prediction.
Combines metrics, correlations, and narrative insights into a readable summary.
"""

import os
import json
import datetime
import pandas as pd
from textwrap import indent
from src.reporting.narrative_builder import build_summary, narrative_from_summary


def generate_report(
    model_dir="models",
    data_path="data/processed/feature_matrix.csv",
    target_col="close",
    feature_prefixes=("ind_", "brent_", "usd_", "news_", "log_"),
    include_metrics=True,
    include_correlations=True,
    include_narrative=True,
    save_markdown=True,
):
    """
    Generate a full summary report after training or prediction.

    Args:
        model_dir (str): Path to saved model artifacts.
        data_path (str): Path to processed feature CSV.
        target_col (str): Target column name (default 'close').
        feature_prefixes (tuple): Prefixes used to detect feature columns.
        include_metrics (bool): Include metrics from JSON log.
        include_correlations (bool): Include correlation summary.
        include_narrative (bool): Include natural-language narrative.
        save_markdown (bool): Save the report as Markdown (.md).

    Returns:
        str: Path to generated report.
    """

    print("🧾 Generating executive report...")

    # --- Load metrics if available ---
    metrics_text = ""
    metrics_path = os.path.join(model_dir, "experiment_log.json")
    if include_metrics and os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        metrics_text = "\n".join([f"- **{k}**: {v:.4f}" if isinstance(v, (float, int)) else f"- **{k}**: {v}" for k, v in metrics.items()])
    else:
        metrics_text = "No metrics file found."

    # --- Load data ---
    df = pd.read_csv(data_path, index_col=0, parse_dates=[0])
