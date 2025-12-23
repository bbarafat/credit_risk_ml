# api/main.py
import json
import joblib
import pandas as pd
from fastapi import FastAPI

from api.schema import CreditApplication, PredictionResponse
from src.metrics import predict_with_threshold

MODEL_PATH = "models/model.joblib"
THRESHOLD_PATH = "models/threshold.json"

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts probability of credit default",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold = json.load(f)["threshold"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication):

    
    data = application.dict()
    df = pd.DataFrame([data])

    
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Predict
    prob = float(model.predict_proba(df)[:, 1][0])
    pred = int(predict_with_threshold([prob], threshold)[0])

    return PredictionResponse(
        default_probability=prob,
        default_prediction=pred,
        threshold=threshold,
    )