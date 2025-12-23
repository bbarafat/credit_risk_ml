import numpy as np
from sklearn.metrics import confusion_matrix

def expected_cost(y_true, y_pred, cost_fp=1, cost_fn=5) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fp * cost_fp + fn * cost_fn)

def predict_with_threshold(y_prob, threshold: float):
    return (y_prob >= threshold).astype(int)

def find_optimal_threshold(y_true, y_prob, thresholds, cost_fp=1, cost_fn=5):
    costs = []
    for t in thresholds:
        y_pred = predict_with_threshold(y_prob, t)
        costs.append(expected_cost(y_true, y_pred, cost_fp, cost_fn))
    i = int(np.argmin(costs))
    return float(thresholds[i]), float(costs[i])