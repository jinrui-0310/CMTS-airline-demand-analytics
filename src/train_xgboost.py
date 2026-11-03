import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor
from data_loader import load_modeling_data


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
    print("Loading XGBoost dataset...", flush=True)

    df = load_modeling_data()

    print(f"Loaded {len(df):,} observations", flush=True)

    train = df[df["Year"] <= 2022]
    val = df[df["Year"] == 2023]
    test = df[df["Year"] >= 2024]

    # Sample training data for faster experimentation
    train = train.sample(
        frac=0.3,
        random_state=42
    )

    print(f"Train: {len(train):,}", flush=True)
    print(f"Validation: {len(val):,}", flush=True)
    print(f"Test: {len(test):,}", flush=True)

    X_train = train[FEATURES].astype("float32")
    y_train = train["target_passengers"].astype("float32")

    X_val = val[FEATURES].astype("float32")
    y_val = val["target_passengers"].astype("float32")

    X_test = test[FEATURES].astype("float32")
    y_test = test["target_passengers"].astype("float32")

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

    print("\nTraining XGBoost...", flush=True)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print("Training completed!", flush=True)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    evaluate("XGBoost Validation", y_val, val_pred)
    evaluate("XGBoost Test", y_test, test_pred)

    evaluate(
        "Seasonal Naive Test",
        y_test,
        test["passengers_lag4"].to_numpy(),
    )

    print("\nFeature Importance:")
    for feature, importance in sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    main()