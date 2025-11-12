"""
src/reporting/narrative_builder.py

Generates human-readable correlation summaries and narrative insights
from model feature correlations and macro relationships.
"""

import pandas as pd
import numpy as np


def build_summary(df, feature_cols, target_col="close"):
    """
    Compute correlations of features vs target, across full history and last 90 days.
    Returns a DataFrame of feature correlation summaries.
    """
    print("🧾 Building correlation summary...")

    corr_all = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
    corr_recent = (
        df[feature_cols + [target_col]]
        .iloc[-90:]
        .corr()[target_col]
        .drop(target_col)
    )

    summary = pd.DataFrame({
        "feature": corr_all.index,
        "corr_all": corr_all.values,
        "corr_last90": corr_recent.reindex(corr_all.index).values
    })

    def categorize_strength(x):
        ax = abs(x)
        if ax >= 0.7: return "High"
        elif ax >= 0.4: return "Moderate"
        elif ax >= 0.2: return "Low"
        else: return "Negligible"

    summary["strength"] = summary["corr_all"].apply(categorize_strength)
    summary["trend_direction"] = np.where(summary["corr_all"] > 0, "Positive", "Negative")
    summary.sort_values("corr_all", ascending=False, inplace=True)
    summary.reset_index(drop=True, inplace=True)

    print(f"✅ Correlation summary computed for {len(summary)} features.")
    return summary


def narrative_from_summary(summary_df):
    """
    Converts a correlation summary DataFrame into a readable executive-style narrative.
    """
    if summary_df.empty:
        return "No correlation insights available."

    lines = []
    lines.append("📊 **Feature Correlation Insights**")
    lines.append("-" * 60)
    top_features = summary_df.head(5)

    for _, row in top_features.iterrows():
        lines.append(
            f"• {row['feature']}: {row['trend_direction']} correlation ({row['strength']}, "
            f"{row['corr_all']:.2f} overall, last 90d {row['corr_last90']:.2f})"
        )

    # general sentiment of correlations
    avg_corr = summary_df["corr_all"].mean()
    pos_ratio = (summary_df["corr_all"] > 0).mean()

    lines.append("\n🧠 **Interpretation:**")
    if pos_ratio > 0.65:
        lines.append("→ Most features show a positive correlation, indicating broad supportive conditions.")
    elif pos_ratio < 0.35:
        lines.append("→ Majority of features are negatively correlated — potential downside bias in near term.")
    else:
        lines.append("→ Mixed correlation landscape — some bullish, some bearish factors.")

    if avg_corr > 0.3:
        lines.append(f"→ Average correlation is moderately positive ({avg_corr:.2f}). Technicals align with macro tailwinds.")
    elif avg_corr < -0.3:
        lines.append(f"→ Average correlation is moderately negative ({avg_corr:.2f}). Macro factors exert drag on prices.")
    else:
        lines.append(f"→ Average correlation magnitude is mild ({avg_corr:.2f}). Price drivers are relatively balanced.")

    return "\n".join(lines)
