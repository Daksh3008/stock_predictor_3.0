"""
src/eval/evaluator.py

Evaluation helpers: regression metrics, directional accuracy, and
prediction interval coverage for uncertainty-enabled forecasts.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression_metrics(y_true, y_pred):
    """
    Compute basic regression metrics.
    y_true, y_pred: 1d arrays
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

def directional_accuracy(y_true, y_pred):
    """
    Directional accuracy: compare sign of day-to-day changes.
    If input are prices, we compute diff first; if inputs are returns, they can be used directly.
    Returns fraction of times direction matches.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) < 2 or len(y_pred) < 2:
        return float('nan')
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    # align lengths
    n = min(len(true_dir), len(pred_dir))
    acc = (true_dir[:n] == pred_dir[:n]).mean()
    return float(acc)

def interval_coverage(y_true, lower, upper):
    """
    Compute empirical coverage: fraction of true values inside [lower, upper].
    y_true, lower, upper should be broadcastable to same shape.
    """
    y_true = np.asarray(y_true).ravel()
    lower = np.asarray(lower).ravel()
    upper = np.asarray(upper).ravel()
    assert len(y_true) == len(lower) == len(upper), "Shapes must match"
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))

def sharpness(std_array):
    """
    Sharpness: mean predictive standard deviation (lower is sharper/less uncertain).
    """
    return float(np.mean(std_array))
