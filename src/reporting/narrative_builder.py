"""
src/reporting/narrative_builder.py

Generates human-readable correlation summaries and narrative insights
from model feature correlations and macro relationships.
"""

import pandas as pd
import numpy as np


import numpy as np
import pandas as pd

def build_summary(df, feature_cols, target_col="close", recent_window=90):
    """
    Computes correlation of each feature with the target variable
    and classifies strength + direction.

    Returns a DataFrame with:
      feature | corr_all | corr_recent | strength | direction
    """

    records = []
    df = df.copy().dropna(subset=[target_col])

    for feat in feature_cols:
        if feat not in df.columns:
            continue

        # Compute correlations (global + recent window)
        try:
            corr_all = df[feat].corr(df[target_col])
            corr_recent = df[feat].iloc[-recent_window:].corr(df[target_col].iloc[-recent_window:])
        except Exception:
            corr_all, corr_recent = np.nan, np.nan

        records.append({
            "feature": feat,
            "corr_all": corr_all,
            "corr_recent": corr_recent,
        })

    corr_df = pd.DataFrame(records).dropna(subset=["corr_all"])

    # --- Add direction and qualitative strength ---
    corr_df["direction"] = np.where(corr_df["corr_all"] >= 0, "Positive", "Negative")
    corr_df["strength"] = pd.cut(
        corr_df["corr_all"].abs(),
        bins=[0, 0.3, 0.6, 1],
        labels=["Low", "Moderate", "High"]
    )

    # Sort by absolute correlation
    corr_df = corr_df.sort_values("corr_all", ascending=False).reset_index(drop=True)

    return corr_df

#-----------------------------------------------------------------------------------------------------------------------
def narrative_from_summary(corr_df, macro_df=None, feature_df=None, prediction=None):
    """
    Builds a full executive-style textual report with macro, micro, and economic context.

    Args:
        corr_df (pd.DataFrame): correlation summary from build_summary()
        macro_df (pd.DataFrame, optional): latest macro variables like Brent, USD/INR, CPI
        feature_df (pd.DataFrame, optional): processed stock-level indicators
        prediction (dict, optional): {'mean': float, 'lower': float, 'upper': float}
    """
    lines = []
    lines.append("📊 **Feature Correlation Insights**")
    lines.append("------------------------------------------------------------")
    for _, row in corr_df.head(8).iterrows():
        lines.append(f"• {row['feature']}: {row['direction']} correlation ({row['strength']}, {row['corr_all']:.2f} overall, last 90d {row['corr_recent']:.2f})")

    lines.append("\n🧠 **Interpretation:**")
    lines.append("→ Correlation strength indicates which features have historically aligned with price movements.")
    lines.append("→ Indicators with strong positive correlation typically reinforce trend persistence.")
    lines.append("→ Negative correlations imply mean-reversion or risk sentiment turning points.\n")

    # --- Macro Context ---
    lines.append("🌍 **Macro Context**")
    lines.append("------------------------------------------------------------")
    lines.append("Brent crude and USD/INR movements are key macro drivers for Deepak Nitrite.")
    lines.append("• Higher crude prices raise raw material costs for chemical intermediates.")
    lines.append("• INR appreciation lowers import costs and improves profit margins.")
    if macro_df is not None:
        brent = macro_df['brent_close'].iloc[-1] if 'brent_close' in macro_df else None
        fx = macro_df['usd_inr'].iloc[-1] if 'usd_inr' in macro_df else None
        if brent:
            lines.append(f"→ Latest Brent close: ${brent:.2f}/bbl")
        if fx:
            lines.append(f"→ Latest USD/INR: {fx:.2f}")
    lines.append("Overall, macro indicators suggest moderate input-cost stability.\n")

    # --- Micro (Technical) Context ---
    lines.append("📈 **Micro / Technical Context**")
    lines.append("------------------------------------------------------------")
    lines.append("Recent price momentum, short-term MAs, and RSI highlight near-term directionality.")
    if feature_df is not None:
        rsi = feature_df['ind_rsi_14'].iloc[-1] if 'ind_rsi_14' in feature_df else None
        ma10 = feature_df['ind_ma10'].iloc[-1] if 'ind_ma10' in feature_df else None
        ma50 = feature_df['ind_ma50'].iloc[-1] if 'ind_ma50' in feature_df else None
        if rsi is not None:
            if rsi < 40:
                lines.append(f"• RSI(14) = {rsi:.1f} → Oversold zone, potential for recovery.")
            elif rsi > 70:
                lines.append(f"• RSI(14) = {rsi:.1f} → Overbought zone, caution advised.")
            else:
                lines.append(f"• RSI(14) = {rsi:.1f} → Neutral to positive momentum.")
        if ma10 and ma50:
            trend = "uptrend" if ma10 > ma50 else "downtrend"
            lines.append(f"• MA10 ({ma10:.2f}) vs MA50 ({ma50:.2f}) → {trend} bias.")
    lines.append("Short-term volatility suggests limited reversal risk if trend sustains.\n")

    # --- Economic Context ---
    lines.append("🏭 **Economic Context**")
    lines.append("------------------------------------------------------------")
    lines.append("Chemical margins remain sensitive to inflation, crude, and industrial demand.")
    lines.append("Stable crude and currency levels imply margin stability in coming weeks.")
    lines.append("Exports outlook depends on China and European industrial restocking cycles.\n")

    # --- Quantitative Forecast Summary ---
    if prediction:
        mean = prediction['mean']
        lower = prediction['lower']
        upper = prediction['upper']
        lines.append("💹 **Forecast Summary**")
        lines.append("------------------------------------------------------------")
        lines.append(f"Predicted price: ₹{mean:,.2f}")
        lines.append(f"95% confidence range: ₹{lower:,.2f} – ₹{upper:,.2f}")
        lines.append("Interpretation: Base case expects steady appreciation under stable macro backdrop.\n")

    return "\n".join(lines)


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
