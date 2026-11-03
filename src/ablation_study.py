"""Ablation study using DuckDB."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from data_loader import load_modeling_data


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_A = [
    "passengers_lag1",
    "passengers_lag4",
    "Quarter",
]

FEATURES_B = [
    "passengers_lag1",
    "passengers_lag4",
    "fare_lag1",
    "hhi_lag1",
    "carriers_lag1",
    "Quarter",
]


def evaluate(name, y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)

    return {
        "Model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    print("Loading ablation dataset from DuckDB...", flush=True)

    df = load_modeling_data()
    df = df[df["Year"] <= 2023].copy()

    print(f"Loaded {len(df):,} observations", flush=True)

    train = df[df["Year"] <= 2022].copy()
    val = df[df["Year"] == 2023].copy()

    train = train.sample(frac=0.10, random_state=42)

    print(f"Train: {len(train):,}", flush=True)
    print(f"Validation: {len(val):,}", flush=True)

    results = []

    for name, features in [
        ("Demand History Only", FEATURES_A),
        ("Demand + Market Conditions", FEATURES_B),
    ]:
        print(f"\nTraining: {name}", flush=True)

        X_train = train[features].astype("float32")
        X_val = val[features].astype("float32")

        y_train = train["target_passengers"].astype("float32")
        y_val = val["target_passengers"].astype("float32")

        medians = X_train.median().fillna(0.0)

        X_train = X_train.fillna(medians)
        X_val = X_val.fillna(medians)

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=10,
            max_features=0.8,
            n_jobs=4,
            random_state=42,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        result = evaluate(name, y_val, predictions)

        results.append(result)

        print(
            f"MAE: {result['MAE']:.2f} | "
            f"RMSE: {result['RMSE']:.2f} | "
            f"R²: {result['R2']:.4f}"
        )

    results_df = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "ablation_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nAblation Study Results")
    print(results_df.to_string(index=False))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()