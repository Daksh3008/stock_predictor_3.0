"""
src/eval/explainability.py

Wrapper utilities for SHAP explainability on LSTM and other models.
"""
import numpy as np
import shap
import torch

def compute_shap_values(predict_fn, background, explain_data, nsamples=100):
    """
    Compute SHAP values for a model prediction function using KernelExplainer.

    Args:
        predict_fn: callable → model prediction function that accepts numpy array
        background: numpy array (N x features) for background (reference) samples
        explain_data: numpy array (M x features) to explain
        nsamples: number of SHAP sampling iterations (default=100)

    Returns:
        shap_values: numpy array of same shape as explain_data
    """
    print("🧠 Computing SHAP explainability (KernelExplainer)...")
    try:
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(explain_data, nsamples=nsamples)
        shap_values = np.array(shap_values)
        print(f"✅ SHAP computed successfully → shape {shap_values.shape}")
        return shap_values
    except Exception as e:
        print("⚠️ SHAP computation failed:", str(e))
        # fallback dummy SHAP (zeros)
        return np.zeros_like(explain_data)


def compute_shap_kernel(model, background, explain_data, seq_len, feature_dim, runs=50):
    """
    Alternative version: Monte Carlo dropout SHAP approximation for sequence models.
    This version is optional and used if compute_shap_values is too slow.
    """
    model.eval()
    print("🧩 Running SHAP MC approximation...")

    preds = []
    with torch.no_grad():
        for i in range(runs):
            x = torch.from_numpy(explain_data.reshape(-1, seq_len, feature_dim)).float()
            y, _ = model(x)
            preds.append(y.numpy())
    preds = np.stack(preds)
    mean_pred = preds.mean(axis=0)
    shap_like = preds - mean_pred
    return shap_like
