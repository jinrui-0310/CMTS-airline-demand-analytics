import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_loader import load_modeling_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES = [
    "passengers_lag1",
    "passengers_lag4",
    "fare_lag1",
    "hhi_lag1",
    "carriers_lag1",
    "route_distance",
    "nonstop_distance",
    "Quarter",
]


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("-" * 35)
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")


def main():

    print("Loading modeling dataset...", flush=True)

    df = load_modeling_data()

    print(f"Loaded {len(df):,} observations", flush=True)

    # Time-based split
    train = df[df["Year"] <= 2022]
    val = df[df["Year"] == 2023]
    test = df[df["Year"] >= 2024]

    print(f"Train: {len(train):,}", flush=True)
    print(f"Validation: {len(val):,}", flush=True)
    print(f"Test: {len(test):,}", flush=True)

    X_train = train[FEATURES]
    y_train = train["target_passengers"]

    X_val = val[FEATURES]
    y_val = val["target_passengers"]

    X_test = test[FEATURES]
    y_test = test["target_passengers"]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("elasticnet", ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            max_iter=5000,
            random_state=42
        ))
    ])

    print("\nTraining ElasticNet...", flush=True)

    model.fit(X_train, y_train)

    print("Training completed!", flush=True)

    val_pred = np.maximum(model.predict(X_val), 0)
    test_pred = np.maximum(model.predict(X_test), 0)

    evaluate("ElasticNet Validation", y_val, val_pred)
    evaluate("ElasticNet Test", y_test, test_pred)

    baseline_pred = test["passengers_lag4"].to_numpy()

    evaluate(
        "Seasonal Naive Test",
        y_test,
        baseline_pred
    )


if __name__ == "__main__":
    main()