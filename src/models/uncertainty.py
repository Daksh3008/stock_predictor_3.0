"""
src/models/uncertainty.py

MC-Dropout wrapper utilities for PyTorch models to obtain predictive
mean and intervals (approximate Bayesian uncertainty).
"""
import numpy as np
import torch

def enable_dropout(model):
    """Enable dropout layers during inference (keeps other layers unchanged)."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout") or isinstance(m, torch.nn.Dropout):
            m.train()

def predict_mc_dropout(model, x_tensor, runs=50, device=None):
    """
    Run forward passes with dropout enabled to estimate predictive distribution.

    Args:
        model: PyTorch model (must return (preds, attn) or preds)
        x_tensor: torch.Tensor of shape (batch, seq_len, features)
        runs: number of stochastic forward passes
        device: 'cpu' or 'cuda'

    Returns:
        dict with keys: mean (np.array), std (np.array), all (np.array of shape runs x batch)
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    # enable dropout layers
    enable_dropout(model)

    x = x_tensor.to(device)
    preds = []
    with torch.no_grad():
        for _ in range(runs):
            out = model(x)
            # model may return tuple (pred, attn)
            if isinstance(out, tuple) or isinstance(out, list):
                p = out[0]
            else:
                p = out
            preds.append(p.detach().cpu().numpy())
    preds = np.stack(preds, axis=0)  # runs x batch
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return {"mean": mean, "std": std, "all": preds}

def predictive_interval(mean, std, z=1.96):
    """
    Approximate symmetric prediction interval around mean using Gaussian approx.
    z=1.96 approx 95% CI.
    """
    return mean - z * std, mean + z * std
