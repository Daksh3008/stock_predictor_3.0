"""Enhanced experiment runner with automatic logging, metrics tracking, and explainability hooks.
This version extends the previous one by adding evaluation metrics, SHAP analysis (optional),
and generation of a correlation-based summary report after training.
"""
import os
import json
import torch
from torch import optim
from src.datasets.timeseries_dataset import TimeSeriesDataset
from src.models.lstm_model import LSTMForecast
from src.train.trainer import Trainer
from sklearn.preprocessing import StandardScaler
from src.eval.evaluator import regression_metrics, directional_accuracy
from src.eval.explainability import compute_shap_values
from src.reporting.narrative_builder import build_summary, narrative_from_summary
import pandas as pd
import numpy as np


def run_experiment(
    processed_csv='data/processed/feature_matrix.csv',
    feature_cols=None,
    target_col='close',
    seq_len=60,
    pred_horizon=1,
    hidden=128,
    layers=2,
    dropout=0.2,
    use_attention=True,
    epochs=20,
    batch=64,
    lr=1e-3,
    explain=True,
    save_dir='models',
    shap_sample=200
):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'experiment_log.json')

    df = pd.read_csv(processed_csv, index_col=0, parse_dates=[0])
    df.columns = [c.lower() for c in df.columns]

    if feature_cols is None:
        feature_cols = [
            c for c in df.columns if c.startswith('ind_') or c in ['brent_close', 'usd_inr', 'news_compound', 'log_return']
        ]

    # Split time-based train/val
    split = int(len(df) * 0.8)
    df_train, df_val = df.iloc[:split], df.iloc[split:]

    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)

    train_ds = TimeSeriesDataset(df_train, feature_cols, target_col, seq_len, pred_horizon, scaler)
    val_ds = TimeSeriesDataset(df_val, feature_cols, target_col, seq_len, pred_horizon, scaler)

    # save fitted feature scaler for later use 
    from src.utils.io import save_scaler
    save_scaler(scaler, os.path.join(save_dir, 'feature_scaler.joblib'))
    scaler_path = os.path.join(save_dir, 'feature_scaler.joblib')

    model = LSTMForecast(
        input_dim=len(feature_cols),
        hidden_dim=hidden,
        num_layers=layers,
        dropout=dropout,
        use_attention=use_attention,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer)

    ckpt_path = os.path.join(save_dir, 'best.pt')
    trainer.fit(train_ds, val_ds, epochs=epochs, batch_size=batch, ckpt_path=ckpt_path)

    # Evaluate on validation set
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model_state'])
    model.eval()

    X_val, y_val = [], []
    for X, y in val_ds:
        X_val.append(X)
        y_val.append(y)
    X_val = np.stack(X_val)
    y_val = np.array(y_val)

    with torch.no_grad():
        preds = []
        for i in range(0, len(X_val), batch):
            xb = torch.from_numpy(X_val[i:i+batch])
            yb, _ = model(xb)
            preds.extend(yb.numpy())
    preds = np.array(preds)

    metrics = regression_metrics(y_val, preds)
    metrics['directional_accuracy'] = directional_accuracy(y_val, preds)

    # Save metrics
    with open(log_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {log_path}\n{metrics}")

    # Optional SHAP explainability
    if explain:
        print("Computing SHAP values (subset)...")
        sample_idx = np.random.choice(len(X_val), min(shap_sample, len(X_val)), replace=False)
        X_bg = X_val[:100].reshape(100, -1)
        X_exp = X_val[sample_idx].reshape(len(sample_idx), -1)
        def predict_fn(Xflat):
            Xseq = torch.from_numpy(Xflat.reshape(-1, seq_len, len(feature_cols))).float()
            with torch.no_grad():
                out, _ = model(Xseq)
                return out.numpy().flatten()
        shap_vals = compute_shap_values(predict_fn, X_bg, X_exp)
        np.save(os.path.join(save_dir, 'shap_values.npy'), shap_vals)

    # Build correlation summary
    corr_summary = build_summary(df, feature_cols, target_col)
    report_text = narrative_from_summary(corr_summary)
    report_path = os.path.join(save_dir, 'feature_correlation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"Saved detailed correlation report → {report_path}")

    return {
        'checkpoint': ckpt_path,
        'metrics': metrics,
        'correlations': corr_summary.to_dict('records'),
        'report': report_path
    }


if __name__ == '__main__':
    result = run_experiment()
    print(json.dumps(result['metrics'], indent=2))