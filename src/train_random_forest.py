"""Train Random Forest using DuckDB instead of SQLite.

Run:
    python src/train_random_forest.py

This script preserves the original training recipe and writes separate
DuckDB artifacts without overwriting SQLite models or predictions.
"""

import argparse
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]

DB = ROOT / "data" / "processed" / "airline_analytics.duckdb"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs" / "tables"

FEATURES = [
    "passengers_lag1",
    "passengers_lag4",
    "fare_lag1",
    "hhi_lag1",
    "carriers_lag1",
    "nonstop_distance",
    "Quarter",
]


def metrics(y, pred):
    pred = np.maximum(np.asarray(pred), 0)

    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction must be in (0, 1]")

    if not DB.is_file():
        raise FileNotFoundError(f"DuckDB database not found: {DB}")

    print("Step 1/5: Reading route_features from DuckDB...", flush=True)

    sql = """
        SELECT
            Year,
            Quarter,
            OriginCityMarketID,
            DestCityMarketID,
            route_id,
            target_passengers,
            passengers_lag1,
            passengers_lag4,
            fare_lag1,
            hhi_lag1,
            carriers_lag1,
            nonstop_distance

        FROM route_features

        WHERE passengers_lag1 IS NOT NULL
          AND passengers_lag4 IS NOT NULL
          AND target_passengers IS NOT NULL
          AND (Year < 2025 OR (Year = 2025 AND Quarter <= 2))

        ORDER BY
            Year,
            Quarter,
            OriginCityMarketID,
            DestCityMarketID,
            route_id
    """

    con = duckdb.connect(str(DB), read_only=True)

    try:
        df = con.execute(sql).df()
    finally:
        con.close()

    if df.empty:
        raise ValueError("No eligible rows found in route_features")

    print(f"Loaded {len(df):,} eligible rows.")

    print("Step 2/5: Splitting datasets...", flush=True)

    train = df.loc[df.Year <= 2022]
    val = df.loc[df.Year == 2023]
    test = df.loc[
        (df.Year >= 2024)
        & ((df.Year < 2025) | (df.Quarter <= 2))
    ]

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Train, validation and test partitions must all be nonempty")

    if args.train_fraction < 1:
        train = train.sample(
            frac=args.train_fraction,
            random_state=42,
        )

    print(
        f"Train: {len(train):,} | "
        f"Validation: {len(val):,} | "
        f"Test: {len(test):,}"
    )

    print("Step 3/5: Preparing features...", flush=True)

    X_train = train[FEATURES].astype("float32")

    medians = X_train.median().fillna(0.0)

    X_train = X_train.fillna(medians)
    X_val = val[FEATURES].astype("float32").fillna(medians)
    X_test = test[FEATURES].astype("float32").fillna(medians)

    y_train = train.target_passengers.to_numpy(dtype="float32")
    y_val = val.target_passengers.to_numpy(dtype="float32")
    y_test = test.target_passengers.to_numpy(dtype="float32")

    print("Step 4/5: Training Random Forest...", flush=True)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=10,
        max_features=0.8,
        n_jobs=args.n_jobs,
        random_state=42,
    )

    model.fit(X_train, y_train)

    val_pred = np.maximum(model.predict(X_val), 0)
    test_pred = np.maximum(model.predict(X_test), 0)

    naive_pred = test.passengers_lag4.to_numpy(dtype="float32")

    results = pd.DataFrame([
        {
            "Model": "Random Forest",
            "Split": "Validation 2023",
            **metrics(y_val, val_pred),
        },
        {
            "Model": "Random Forest",
            "Split": "Test 2024Q1–2025Q2",
            **metrics(y_test, test_pred),
        },
        {
            "Model": "Seasonal Naive",
            "Split": "Test 2024Q1–2025Q2",
            **metrics(y_test, naive_pred),
        },
    ])

    print()
    print(results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))

    print("Step 5/5: Saving DuckDB artifacts...", flush=True)

    predictions = test[[
        "Year",
        "Quarter",
        "OriginCityMarketID",
        "DestCityMarketID",
        "route_id",
        "target_passengers",
        "passengers_lag4",
    ]].copy()

    predictions["rf_predicted"] = test_pred
    predictions["naive_predicted"] = naive_pred

    predictions["rf_error"] = (
        predictions.target_passengers - predictions.rf_predicted
    )

    predictions["naive_error"] = (
        predictions.target_passengers - predictions.naive_predicted
    )

    predictions["rf_abs_error"] = predictions.rf_error.abs()
    predictions["naive_abs_error"] = predictions.naive_error.abs()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "random_forest.joblib"

    pred_path = (
        OUTPUT_DIR / "random_forest_predictions.csv"
    )

    metrics_path = (
        OUTPUT_DIR / "random_forest_metrics.csv"
    )

    config_path = (
        OUTPUT_DIR / "random_forest_training_config.json"
    )

    joblib.dump(
        {
            "model": model,
            "medians": medians,
            "features": FEATURES,
        },
        model_path,
    )

    predictions.to_csv(pred_path, index=False)
    results.to_csv(metrics_path, index=False)

    config = {
        "database": "DuckDB",
        "source_table": "route_features",
        "training_years": "<=2022",
        "validation_year": 2023,
        "test_period": "2024Q1–2025Q2",
        "train_fraction": args.train_fraction,
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
        "features": FEATURES,
        "model_params": model.get_params(),
        "note": (
            "DuckDB migration of train_random_forest_final.py. "
            "Uses the same training recipe."
        ),
    }

    config_path.write_text(
        json.dumps(config, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nSaved model: {model_path}")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved configuration: {config_path}")

    print("\nOriginal SQLite artifacts were NOT overwritten.")


if __name__ == "__main__":
    main()