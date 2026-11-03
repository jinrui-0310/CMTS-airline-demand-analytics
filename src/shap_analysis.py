"""SHAP analysis for the DuckDB Random Forest model."""

from pathlib import Path

import duckdb
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "data/processed/airline_analytics.duckdb"
MODEL_PATH = ROOT / "models/random_forest.joblib"

OUTPUT_DIR = ROOT / "outputs/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    # --------------------------------------------------------
    # 1. Load model
    # --------------------------------------------------------

    print("Step 1/5: Loading DuckDB Random Forest model...", flush=True)

    saved = joblib.load(MODEL_PATH)

    model = saved["model"]
    features = saved["features"]
    medians = saved["medians"]

    print("Model loaded!", flush=True)

    # --------------------------------------------------------
    # 2. Load validation data
    # --------------------------------------------------------

    print("Step 2/5: Loading validation data from DuckDB...", flush=True)

    columns = ", ".join(features)

    query = f"""
        SELECT {columns}
        FROM route_features
        WHERE Year = 2023
          AND passengers_lag1 IS NOT NULL
          AND passengers_lag4 IS NOT NULL
        ORDER BY
            Year,
            Quarter,
            OriginCityMarketID,
            DestCityMarketID,
            route_id
    """

    con = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        df = con.execute(query).df()
    finally:
        con.close()

    # Reproducible sampling
    df = df.sample(
        n=min(1000, len(df)),
        random_state=42
    )

    X = df[features].astype("float32")
    X = X.fillna(medians)

    print(f"SHAP sample size: {len(X):,}", flush=True)

    # --------------------------------------------------------
    # 3. Calculate SHAP
    # --------------------------------------------------------

    print("Step 3/5: Calculating SHAP values...", flush=True)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X)

    print("SHAP calculation completed!", flush=True)

    # --------------------------------------------------------
    # 4. Generate plots
    # --------------------------------------------------------

    print("Step 4/5: Generating SHAP figures...", flush=True)

    # Beeswarm
    shap.plots.beeswarm(
        shap_values,
        show=False,
        max_display=len(features)
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_beeswarm.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Feature importance
    shap.plots.bar(
        shap_values,
        show=False,
        max_display=len(features)
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_feature_importance.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # Waterfall
    shap.plots.waterfall(
        shap_values[0],
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_waterfall.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # 5. Export numerical SHAP importance
    # --------------------------------------------------------

    print("Step 5/5: Exporting SHAP importance...", flush=True)

    importance = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    }).sort_values(
        "mean_abs_shap",
        ascending=False
    )

    importance_path = (
        ROOT / "outputs/tables/shap_importance.csv"
    )

    importance_path.parent.mkdir(parents=True, exist_ok=True)

    importance.to_csv(
        importance_path,
        index=False
    )

    print("\nSHAP Feature Importance:")
    print(importance.to_string(index=False))

    print("\nSHAP analysis completed!")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print(f"Importance saved to: {importance_path}")


if __name__ == "__main__":
    main()