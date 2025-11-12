"""
src/pipeline/predict.py

Loads trained LSTM + Attention model and scaler, runs batch prediction on
the most recent window, and prints a formatted executive report.

Usage:
    python -m src.pipeline.predict --horizon 20
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from src.models.lstm_model import LSTMForecast
from src.models.uncertainty import predict_mc_dropout, predictive_interval  # ✅ matches your actual filename
from src.utils.io import load_scaler
from src.reporting.narrative_builder import build_summary, narrative_from_summary


# ✅ Correct paths relative to project structure
DATA_PATH = os.path.join("data", "processed", "feature_matrix.csv")
MODEL_PATH = os.path.join("models", "best.pt")
SCALER_PATH = os.path.join("models", "feature_scaler.joblib")  # ✅ Corrected filename


def main():
    parser = argparse.ArgumentParser(description="Generate price prediction with uncertainty")
    parser.add_argument("--horizon", type=int, default=20, help="Days ahead to predict")
    args = parser.parse_args()

    print("=" * 80)
    print(f"🔮 Predicting Deepak Nitrite price {args.horizon} days ahead")
    print("=" * 80)

    # ✅ Load data and lowercase all column names for consistency
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=[0])
    df.columns = [c.lower() for c in df.columns]

    # ✅ Ensure feature consistency
    feature_cols = [
        c for c in df.columns
        if c.startswith("ind_") or c in ["brent_close", "usd_inr", "news_compound", "log_return"]
    ]

    # ✅ Load trained scaler
    scaler = load_scaler(SCALER_PATH)

    # ✅ Prepare the last 60 timesteps as model input
    seq_len = 60
    X = df[feature_cols].values[-seq_len:]
    X_scaled = scaler.transform(X)
    X_scaled = np.expand_dims(X_scaled, axis=0).astype(np.float32)

    # ✅ Load trained model (ensure input_dim consistency)
    model = LSTMForecast(input_dim=len(feature_cols))
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # ✅ Monte Carlo dropout predictions (uncertainty estimation)
    x_tensor = torch.from_numpy(X_scaled)
    mc = predict_mc_dropout(model, x_tensor, runs=50)
    mean_pred = mc["mean"].item()
    std_pred = mc["std"].item()
    lower, upper = predictive_interval(mean_pred, std_pred)

    # ✅ Context summary (lowercase target col)
    corr_summary = build_summary(df, feature_cols, target_col="close")
    macro_df = df[['brent_close','usd_inr']].dropna()
    feature_df = df[feature_cols]
    prediction_info = {'mean': mean_pred, 'lower': lower, 'upper': upper}

    narrative = narrative_from_summary(
        corr_summary, macro_df=macro_df, feature_df=feature_df, prediction=prediction_info
    )

    # ✅ Print final executive-style report
    print(f"📈 Predicted Price: ₹{mean_pred:,.2f}")
    print(f"Confidence (±1.96σ): ±{1.96 * std_pred:,.2f} → Range ₹{lower:,.2f} – ₹{upper:,.2f}")
    print("\nMacro & Technical Insights:")
    for _, row in corr_summary.head(5).iterrows():
        print(f"  • {row['feature']} → {row['corr_all']:.2f} ({row['strength']})")

    print("\nNarrative Summary:")
    print(narrative)
    print("=" * 80)


if __name__ == "__main__":
    main()
