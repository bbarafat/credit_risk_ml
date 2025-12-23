import json
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict

from src.config import TARGET, ID_COL, COST_FP, COST_FN, RANDOM_STATE, TEST_SIZE, N_SPLITS, THRESH_GRID
from src.features import build_preprocessor
from src.metrics import find_optimal_threshold


def main(input_csv: str, model_dir: str):
    df = pd.read_csv(input_csv)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    
    pay_cols = [f"X{i}" for i in range(6, 12)]
    education_cols = ["X3"]
    marital_cols = ["X4"]
    categorical_features = ["X2"]
    continuous_features = (["X1"] + [f"X{i}" for i in range(12, 24)] + ["X5"])

    preprocessor = build_preprocessor(
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        pay_cols=pay_cols,
        education_cols=education_cols,
        marital_cols=marital_cols,
    )

    from sklearn.ensemble import HistGradientBoostingClassifier
    estimator = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        random_state=RANDOM_STATE)

    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = cross_val_predict(
        pipeline, X_train_full, y_train_full, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    threshold, _ = find_optimal_threshold(
        y_true=y_train_full,
        y_prob=oof_prob,
        thresholds=np.array(THRESH_GRID),
        cost_fp=COST_FP,
        cost_fn=COST_FN,
    )

    pipeline.fit(X_train_full, y_train_full)

    # Save model
    joblib.dump(pipeline, f"{model_dir}/model.joblib")
    with open(f"{model_dir}/threshold.json", "w") as f:
        json.dump({"threshold": threshold, "cost_fp": COST_FP, "cost_fn": COST_FN}, f)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--model_dir", required=True)
    args = p.parse_args()
    main(args.input_csv, args.model_dir)