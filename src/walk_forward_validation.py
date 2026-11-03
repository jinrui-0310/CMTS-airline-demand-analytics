"""Walk-forward validation using DuckDB."""

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

FEATURES = [
    "passengers_lag1",
    "passengers_lag4",
    "fare_lag1",
    "hhi_lag1",
    "carriers_lag1",
    "Quarter",
]

VALIDATION_YEARS = [2019, 2020, 2021, 2022, 2023]


def evaluate(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    print("Loading walk-forward dataset from DuckDB...", flush=True)

    df = load_modeling_data()
    df = df[df["Year"] <= 2023].copy()

    print(f"Loaded {len(df):,} observations", flush=True)

    results = []

    for validation_year in VALIDATION_YEARS:
        print(
            f"\n========== Validation Year: {validation_year} ==========",
            flush=True,
        )

        train = df[df["Year"] < validation_year].copy()
        val = df[df["Year"] == validation_year].copy()

        train = train.sample(frac=0.10, random_state=42)

        print(f"Train: {len(train):,}", flush=True)
        print(f"Validation: {len(val):,}", flush=True)

        X_train = train[FEATURES].astype("float32")
        X_val = val[FEATURES].astype("float32")

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

        print("Training Random Forest...", flush=True)
        model.fit(X_train, y_train)

        rf_pred = model.predict(X_val)
        naive_pred = val["passengers_lag4"].to_numpy()

        rf_metrics = evaluate(y_val, rf_pred)
        naive_metrics = evaluate(y_val, naive_pred)

        results.append({
            "Validation_Year": validation_year,
            "Train_Size": len(train),
            "Validation_Size": len(val),
            "RF_MAE": rf_metrics["MAE"],
            "Naive_MAE": naive_metrics["MAE"],
            "RF_RMSE": rf_metrics["RMSE"],
            "Naive_RMSE": naive_metrics["RMSE"],
            "RF_R2": rf_metrics["R2"],
            "Naive_R2": naive_metrics["R2"],
        })

        print(
            f"RF MAE: {rf_metrics['MAE']:.2f} | "
            f"Naive MAE: {naive_metrics['MAE']:.2f}"
        )

        print(
            f"RF RMSE: {rf_metrics['RMSE']:.2f} | "
            f"Naive RMSE: {naive_metrics['RMSE']:.2f}"
        )

    results_df = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "walk_forward_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nWalk-Forward Validation Results")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()