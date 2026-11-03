import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "random_forest_predictions.csv"
)

OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load predictions
# ============================================================

print("Loading prediction results...", flush=True)

df = pd.read_csv(INPUT_PATH)

print(f"Loaded {len(df):,} observations", flush=True)


# ============================================================
# 3. Calculate overall error metrics
# ============================================================

def calculate_metrics(group):

    actual = group["target_passengers"].to_numpy()

    rf_error = group["rf_error"].to_numpy()
    naive_error = group["naive_error"].to_numpy()

    return pd.Series({
        "n_observations": len(group),

        "rf_mae": np.mean(np.abs(rf_error)),
        "naive_mae": np.mean(np.abs(naive_error)),

        "rf_rmse": np.sqrt(np.mean(rf_error ** 2)),
        "naive_rmse": np.sqrt(np.mean(naive_error ** 2)),

        "rf_median_ae": np.median(np.abs(rf_error)),
        "naive_median_ae": np.median(np.abs(naive_error)),

        "rf_p90_ae": np.percentile(np.abs(rf_error), 90),
        "naive_p90_ae": np.percentile(np.abs(naive_error), 90),

        "rf_p99_ae": np.percentile(np.abs(rf_error), 99),
        "naive_p99_ae": np.percentile(np.abs(naive_error), 99),
    })


overall = calculate_metrics(df)

print("\nOverall Error Analysis")
print("-" * 50)
print(overall.round(2))

overall.to_csv(
    OUTPUT_TABLE_DIR / "overall_error_metrics.csv"
)


# ============================================================
# 4. Error analysis by year
# ============================================================

yearly = pd.DataFrame([
    {"Year": year, **calculate_metrics(group).to_dict()}
    for year, group in df.groupby("Year")
])

print("\nError Analysis by Year")
print("-" * 50)

print(
    yearly[
        ["Year", "n_observations",
         "rf_mae", "naive_mae",
         "rf_rmse", "naive_rmse"]
    ].round(2).to_string(index=False)
)

yearly.to_csv(
    OUTPUT_TABLE_DIR / "error_by_year.csv",
    index=False
)


# ============================================================
# 5. Error analysis by route size
# ============================================================

# Use previous year's same-quarter passenger count
# to define route size, avoiding target-based grouping.

bins = [-np.inf, 10, 50, 200, 1000, np.inf]

labels = [
    "Very Small (<=10)",
    "Small (11-50)",
    "Medium (51-200)",
    "Large (201-1000)",
    "Very Large (>1000)",
]

df["route_size"] = pd.cut(
    df["passengers_lag4"],
    bins=bins,
    labels=labels
)

route_size = pd.DataFrame([
    {"route_size": str(size), **calculate_metrics(group).to_dict()}
    for size, group in df.groupby("route_size", observed=True)
])

print("\nError Analysis by Route Size")
print("-" * 50)

print(
    route_size[
        ["route_size", "n_observations",
         "rf_mae", "naive_mae",
         "rf_rmse", "naive_rmse"]
    ].round(2).to_string(index=False)
)

route_size.to_csv(
    OUTPUT_TABLE_DIR / "error_by_route_size.csv",
    index=False
)


# ============================================================
# 6. Plot: MAE by route size
# ============================================================

x = np.arange(len(route_size))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    x - width / 2,
    route_size["rf_mae"],
    width,
    label="Random Forest"
)

ax.bar(
    x + width / 2,
    route_size["naive_mae"],
    width,
    label="Seasonal Naive"
)

ax.set_xticks(x)
ax.set_xticklabels(
    route_size["route_size"],
    rotation=20,
    ha="right"
)

ax.set_ylabel("Mean Absolute Error")
ax.set_title("Forecast MAE by Route Size")
ax.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE_DIR / "mae_by_route_size.png",
    dpi=200
)

plt.close()


# ============================================================
# 7. Plot: RMSE by route size
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    x - width / 2,
    route_size["rf_rmse"],
    width,
    label="Random Forest"
)

ax.bar(
    x + width / 2,
    route_size["naive_rmse"],
    width,
    label="Seasonal Naive"
)

ax.set_xticks(x)
ax.set_xticklabels(
    route_size["route_size"],
    rotation=20,
    ha="right"
)

ax.set_ylabel("Root Mean Squared Error")
ax.set_title("Forecast RMSE by Route Size")
ax.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE_DIR / "rmse_by_route_size.png",
    dpi=200
)

plt.close()


# ============================================================
# 8. Plot: MAE by year
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    yearly["Year"],
    yearly["rf_mae"],
    marker="o",
    label="Random Forest"
)

ax.plot(
    yearly["Year"],
    yearly["naive_mae"],
    marker="o",
    label="Seasonal Naive"
)

ax.set_xlabel("Year")
ax.set_ylabel("Mean Absolute Error")
ax.set_title("Forecast MAE by Year")
ax.set_xticks(yearly["Year"])
ax.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE_DIR / "mae_by_year.png",
    dpi=200
)

plt.close()


# ============================================================
# 9. Plot: Absolute error distribution
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Clip extreme errors for visualization only.
# Original errors remain unchanged in all metrics.

rf_errors = np.abs(df["rf_error"])
naive_errors = np.abs(df["naive_error"])

upper = np.percentile(
    np.concatenate([
        rf_errors.to_numpy(),
        naive_errors.to_numpy()
    ]),
    99
)

ax.hist(
    rf_errors.clip(upper=upper),
    bins=60,
    alpha=0.6,
    label="Random Forest"
)

ax.hist(
    naive_errors.clip(upper=upper),
    bins=60,
    alpha=0.6,
    label="Seasonal Naive"
)

ax.set_xlabel("Absolute Prediction Error")
ax.set_ylabel("Number of Observations")
ax.set_title("Absolute Error Distribution (Clipped at P99)")
ax.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE_DIR / "error_distribution.png",
    dpi=200
)

plt.close()


print("\nError analysis completed!")

print(f"Tables saved to: {OUTPUT_TABLE_DIR}")
print(f"Figures saved to: {OUTPUT_FIGURE_DIR}")
