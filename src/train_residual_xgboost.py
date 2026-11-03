import duckdb
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "airline_analytics.duckdb"

FEATURES = [
    "passengers_lag1",
    "passengers_lag4",
    "fare_lag1",
    "hhi_lag1",
    "carriers_lag1",
    "Quarter",
]


def evaluate(name, y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("-" * 35)
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")


def main():

    print("Loading residual modeling dataset...", flush=True)

    conn = duckdb.connect(str(DB_PATH), read_only=True)

    query = """
        SELECT
            Year,
            Quarter,
            target_passengers,
            passengers_lag1,
            passengers_lag4,
            fare_lag1,
            hhi_lag1,
            carriers_lag1
        FROM route_features
        WHERE passengers_lag1 IS NOT NULL
          AND passengers_lag4 IS NOT NULL
    """

    try:
        df = conn.execute(query).df()
    finally:
        conn.close()

    print(f"Loaded {len(df):,} observations", flush=True)

    train = df[df["Year"] <= 2022].copy()
    val = df[df["Year"] == 2023].copy()
    test = df[df["Year"] >= 2024].copy()

    # Use a training subset for fast experimentation
    train = train.sample(frac=0.3, random_state=42)

    print(f"Train: {len(train):,}", flush=True)
    print(f"Validation: {len(val):,}", flush=True)
    print(f"Test: {len(test):,}", flush=True)

    # Residual target
    train["residual"] = (
        train["target_passengers"]
        - train["passengers_lag4"]
    )

    X_train = train[FEATURES].astype("float32")
    y_train = train["residual"].astype("float32")

    X_val = val[FEATURES].astype("float32")
    X_test = test[FEATURES].astype("float32")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )

    print("\nTraining Residual XGBoost...", flush=True)

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    print("Training completed!", flush=True)

    # Predict the adjustment to last year's demand
    val_residual_pred = model.predict(X_val)
    test_residual_pred = model.predict(X_test)

    # Final forecast = seasonal naive + predicted adjustment
    val_pred = (
        val["passengers_lag4"].to_numpy()
        + val_residual_pred
    )

    test_pred = (
        test["passengers_lag4"].to_numpy()
        + test_residual_pred
    )

    evaluate(
        "Residual XGBoost Validation",
        val["target_passengers"],
        val_pred,
    )

    evaluate(
        "Residual XGBoost Test",
        test["target_passengers"],
        test_pred,
    )

    evaluate(
        "Seasonal Naive Test",
        test["target_passengers"],
        test["passengers_lag4"],
    )


if __name__ == "__main__":
    main()