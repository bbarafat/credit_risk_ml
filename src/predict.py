import json
import joblib
import numpy as np
import pandas as pd
from src.metrics import predict_with_threshold

def load_artifacts(model_path: str, threshold_path: str):
    model = joblib.load(model_path)
    with open(threshold_path, "r") as f:
        threshold = json.load(f)["threshold"]
    return model, float(threshold)

def predict_one(model, threshold: float, row: dict):
    X = pd.DataFrame([row])
    prob = float(model.predict_proba(X)[:, 1][0])
    pred = int(predict_with_threshold(np.array([prob]), threshold)[0])
    return prob, pred