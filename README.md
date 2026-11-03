# U.S. Airline Demand Analytics

A Python-based data engineering and machine learning platform for analyzing U.S. airline markets and forecasting quarterly passenger demand.

The project integrates large-scale airline itinerary data, builds a reproducible analytical pipeline using Python, DuckDB, SQL, and Parquet, and evaluates machine learning models for route-level passenger demand forecasting.

An interactive Streamlit dashboard provides access to historical market trends, route-level analysis, forecasting results, and model explainability.

**Dataset coverage:** 2010 Q1–2025 Q2  
**Dataset size:** 19,948,408 route-quarter observations  
**Test period:** 2024 Q1–2025 Q2

## 1. Project Overview

The project provides an end-to-end analytical workflow for:

- Processing and integrating large-scale airline datasets.
- Analyzing passenger demand, airfare, and market competition.
- Engineering time-aware forecasting features.
- Training and evaluating machine learning models.
- Comparing machine learning predictions against a seasonal baseline.
- Interpreting predictions using SHAP.
- Presenting results through an interactive dashboard.

The implementation emphasizes data engineering, computational analysis, machine learning workflows, and reproducibility.

### Technology Stack

| Component | Technologies |
|---|---|
| Programming | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Analytical Database | DuckDB 2.0.0 |
| Data Storage | CSV, Parquet |
| Query Language | SQL |
| Machine Learning | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |

## 2. System Architecture

    Raw Airline Datasets
             |
             v
    Data Cleaning and Integration
             |
             v
    Parquet Storage + DuckDB
             |
             v
    SQL Analysis and Feature Engineering
             |
             v
    Time-Aware Modeling Dataset
             |
             v
    Machine Learning Training
             |
             v
    Model Evaluation and Explainability
             |
             v
    Streamlit Dashboard

The complete workflow is orchestrated by `run_pipeline.py`.

## 3. Project Structure

    CMTS-airline-demand-analytics/
    ├── app.py/
    │   └── app.py
    ├── src/
    │   ├── data_pipeline.py
    │   ├── build_duckdb.py
    │   ├── build_duckdb_features.py
    │   ├── validate_duckdb_features.py
    │   ├── build_route_mapping.py
    │   ├── export_kpis.py
    │   ├── visualize.py
    │   ├── run_sql_analysis.py
    │   ├── data_loader.py
    │   ├── train_elasticnet.py
    │   ├── train_xgboost.py
    │   ├── train_residual_xgboost.py
    │   ├── train_random_forest.py
    │   ├── analyze_errors.py
    │   ├── walk_forward_validation.py
    │   ├── ablation_study.py
    │   ├── shap_analysis.py
    │   └── dashboard_data_audit.py
    ├── sql/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── models/
    ├── outputs/
    │   ├── tables/
    │   └── figures/
    ├── run_pipeline.py
    ├── requirements.txt
    └── README.md

## 4. Dataset

The project uses U.S. airline itinerary data derived from the Bureau of Transportation Statistics (BTS) DB1B dataset, together with market concentration information.

### Data Coverage

| Dataset | Observations |
|---|---:|
| Integrated route-quarter records | 19,948,408 |
| Eligible modeling records | 8,815,216 |
| Test-set predictions | 992,290 |

The historical dataset covers 2010 Q1 through 2025 Q2.

Passenger volumes represent observations in the processed DB1B dataset rather than a complete census of U.S. airline passengers.

The dataset includes nonstop and connecting itineraries.

## 5. Data Engineering Pipeline

The pipeline consists of four stages.

| Stage | Description |
|---|---|
| Data | Integrates raw datasets, builds DuckDB tables and views, generates features, and exports analytical tables |
| SQL | Runs data validation and market analysis queries |
| Models | Trains ElasticNet, XGBoost, Residual XGBoost, and Random Forest |
| Analysis | Performs error analysis, walk-forward validation, ablation studies, SHAP analysis, and dashboard data auditing |

### Data Integration

The processing workflow includes:

- Loading raw CSV datasets.
- Cleaning and validating observations.
- Constructing route-quarter identifiers.
- Integrating market-level competition features.
- Exporting processed data to Parquet.
- Creating analytical tables and views in DuckDB.

### SQL Analysis

| SQL Script | Purpose |
|---|---|
| `01_data_validation.sql` | Data validation |
| `02_market_analysis.sql` | Market analysis |
| `03_route_features.sql` | Route-level feature analysis |
| `04_route_feature_engineering.sql` | Feature engineering |
| `05_build_route_features.sql` | Feature table construction |
| `06_modeling_dataset.sql` | Modeling dataset analysis |

DuckDB supports analytical queries and feature generation. Parquet provides storage for processed route-quarter data.

### Data Validation

The project validates:

- Route-quarter key uniqueness.
- Temporal alignment of lagged features.
- Market-level feature consistency.
- Prediction record uniqueness.
- Airport mapping consistency.
- Forecast data coverage.

The latest feature validation run found no duplicate keys or invalid lag-1/lag-4 alignments.

## 6. Feature Engineering

The forecasting dataset incorporates historical demand, airfare, competition, route distance, and seasonality.

### Random Forest Features

| Feature | Description |
|---|---|
| `passengers_lag1` | Previous-quarter passenger demand |
| `passengers_lag4` | Passenger demand in the same quarter of the previous year |
| `fare_lag1` | Previous-quarter average airfare |
| `hhi_lag1` | Previous-quarter market concentration |
| `carriers_lag1` | Previous-quarter number of carriers |
| `nonstop_distance` | Direct origin-to-destination distance |
| `Quarter` | Calendar quarter representing seasonality |

### Distance Features

The processed data distinguishes between:

- `nonstop_distance`: Direct origin-to-destination distance.
- `route_distance`: Distance associated with the full itinerary, including connecting segments.

The forecasting models use `nonstop_distance` as a stable route characteristic.

### Seasonality

Seasonality is represented through:

- `Quarter`, identifying the calendar quarter.
- `passengers_lag4`, capturing demand in the same quarter of the previous year.

Lagged features are constructed using historical observations to reduce temporal leakage.

## 7. Forecasting Models

The project implements five forecasting approaches:

1. Seasonal Naive
2. ElasticNet
3. XGBoost
4. Residual XGBoost
5. Random Forest

The dashboard focuses on the primary **Random Forest** model and the **Seasonal Naive** baseline.

Additional candidate models remain available in the training pipeline.

### Time-Based Evaluation

The modeling workflow uses chronological splits.

| Period | Purpose |
|---|---|
| Through 2022 | Training |
| 2023 | Validation |
| 2024 Q1–2025 Q2 | Final testing |

### Seasonal Naive Baseline

The Seasonal Naive model predicts demand using the corresponding quarter from the previous year:

    Prediction(t) = Passenger Demand(t - 4 quarters)

This provides a simple historical benchmark for evaluating machine learning models.

## 8. Model Evaluation

### Final Test-Set Results

The following results are from the latest completed evaluation on the 2024 Q1–2025 Q2 test set.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Random Forest** | **16.4928** | 96.7351 | 0.9788 |
| Seasonal Naive | 16.9854 | **86.4110** | **0.9831** |

Random Forest achieves approximately **2.90% lower MAE** than the Seasonal Naive baseline.

However, Seasonal Naive achieves lower RMSE and higher R².

### Key Findings

- Random Forest produces lower average absolute prediction error.
- Seasonal Naive performs better under metrics that are more sensitive to large prediction errors.
- More complex models do not necessarily outperform a strong seasonal baseline across all metrics.

Other candidate models are implemented in the repository but are not included in the dashboard's focused comparison.

Training configurations and sample sizes differ between candidate models, so the broader comparison should not be interpreted as a fully controlled benchmark.

## 9. Model Analysis

The project includes error analysis, walk-forward validation, ablation studies, SHAP explainability, and dashboard data auditing.

### Error Analysis

Examines prediction errors across different route-demand categories and time periods.

The analysis includes MAE, RMSE, and absolute error distributions.

### Walk-Forward Validation

Expanding-window validation evaluates forecasting performance across successive validation years.

The implementation includes validation periods from 2019 through 2023, covering the COVID-19 disruption and subsequent recovery.

### Ablation Study

An ablation experiment compares two feature groups.

**Demand History Only**

- Previous-quarter passenger demand.
- Same-quarter previous-year demand.
- Calendar quarter.

**Demand + Market Conditions**

- Historical demand and calendar quarter.
- Lagged airfare.
- Lagged market concentration.
- Lagged number of carriers.
- Nonstop distance.

#### Validation Results

| Feature Group | MAE | RMSE | R² |
|---|---:|---:|---:|
| Demand History Only | 16.8466 | 96.6030 | 0.9788 |
| Demand + Market Conditions | **16.4299** | **91.7930** | **0.9809** |

The expanded feature group reduces:

- **MAE by approximately 2.47%.**
- **RMSE by approximately 4.98%.**

These improvements describe the combined contribution of additional features rather than the isolated effect of any single variable.

### SHAP Explainability

SHAP is used to interpret Random Forest predictions.

| Rank | Feature | Mean Absolute SHAP |
|---|---|---:|
| 1 | `passengers_lag1` | 72.7480 |
| 2 | `passengers_lag4` | 21.2001 |
| 3 | `Quarter` | 2.2229 |
| 4 | `fare_lag1` | 1.5022 |
| 5 | `carriers_lag1` | 0.7074 |
| 6 | `hhi_lag1` | 0.6538 |
| 7 | `nonstop_distance` | 0.5371 |

Historical passenger demand provides the strongest predictive information.

Seasonality, airfare, competition, and distance contribute additional information, although their relative importance is smaller.

SHAP values describe model behavior and should not be interpreted as causal effects.

### Dashboard Data Audit

Validates the consistency of data used by the Streamlit dashboard, including prediction uniqueness and airport mappings.

## 10. Interactive Dashboard

The Streamlit dashboard contains four sections.

### Market Overview

Provides:

- Quarterly passenger demand trends.
- Passenger-weighted average airfare.
- Top airport-pair markets.
- COVID-19 demand and airfare recovery analysis.
- Interactive filters and CSV exports.

### Route Explorer

Allows users to select origin and destination airports and examine:

- Historical passenger demand.
- Average airfare.
- Market concentration (HHI).
- Number of carriers.
- Nonstop route distance.

### Demand Forecasting

Displays historical out-of-sample predictions for 2024 Q1–2025 Q2.

Features include:

- Origin and destination selection.
- Individual itinerary and airport-pair aggregate views.
- Random Forest versus Seasonal Naive predictions.
- Actual versus predicted demand.
- Quarterly forecast errors.
- MAE and RMSE comparisons.
- CSV export.

**Important:** These results are historical backtests, not live forecasts of future quarters.

### Model Insights

Provides:

- Model comparison.
- Forecast error analysis.
- Walk-forward validation.
- SHAP feature importance and explanations.
- Ablation study results.

### Launch the Dashboard

From the project root, run:

    python -m streamlit run app.py/app.py

The required processed data and model outputs must be generated before launching the dashboard.

## 11. Installation

### Step 1: Clone the Repository

    git clone <repository-url>
    cd CMTS-airline-demand-analytics

Replace `<repository-url>` with the repository's Git URL.

### Step 2: Create a Python Environment

Python 3.11 was used during development.

Using a virtual environment:

    python3.11 -m venv .venv
    source .venv/bin/activate

Alternatively, using Conda:

    conda create -n cmts python=3.11
    conda activate cmts

Use one environment consistently when installing dependencies and running the pipeline.

### Step 3: Install Dependencies

    python -m pip install -r requirements.txt

The development environment uses DuckDB 2.0.0.

### Step 4: Prepare Source Data

Place the following files in `data/raw/`:

    data/raw/
    ├── DB1B_RoutePanel_Core_2010Q1_2025Q2.csv
    └── HHI_market_quarter_2010Q1_2025Q2.csv

The exact source files are not included in the Git repository because of their size.

A public download location for these prepared inputs is not currently provided.

## 12. Running the Pipeline

### Complete Pipeline

Execute all stages:

    python run_pipeline.py

The workflow runs:

    data → sql → models → analysis

### Individual Stages

    python run_pipeline.py --data
    python run_pipeline.py --sql
    python run_pipeline.py --models
    python run_pipeline.py --analysis

### Multiple Stages

    python run_pipeline.py --models --analysis

### Stage Dependencies

- Data processing must run before SQL analysis and model training.
- Model training requires the processed feature dataset.
- Downstream model analysis requires trained models and prediction outputs.
- The dashboard requires generated database tables, prediction files, and analytical results.

Once the processed database exists, model training and analysis can be rerun without repeating raw-data processing.

## 13. Reproducibility and Limitations

### Reproducibility

The project provides:

- Modular Python scripts.
- SQL-based analytical transformations.
- A pipeline entry point.
- Time-based train/validation/test splits.
- Explicit model training configurations.
- Saved model artifacts and evaluation outputs.
- Automated data consistency checks.

### Current Limitations

- Raw source datasets are not included in the repository.
- Processed Parquet files, DuckDB databases, trained models, and generated outputs are excluded from Git.
- Python dependencies are not fully version-pinned.
- Some DuckDB views reference local Parquet files.
- Training configurations and sample sizes differ between candidate models.
- The dataset contains sampled DB1B itinerary records rather than complete U.S. passenger traffic.
- The dashboard presents historical backtesting results rather than future operational forecasts.
- Forecast accuracy may vary across route sizes and unusual market conditions.

## 14. Future Improvements

Potential extensions include:

- Automated retrieval and processing of newly released airline data.
- Additional economic and operational forecasting features.
- More systematic hyperparameter optimization.
- Standardized training samples for controlled model comparisons.
- Additional forecasting horizons.
- Model monitoring and automated retraining.
- Deployment of the Streamlit dashboard with a managed data backend.

## 15. Conclusion

This project demonstrates an end-to-end computational workflow for large-scale airline market analysis and passenger demand forecasting.

It combines **Python-based data engineering, DuckDB and SQL analytics, time-aware machine learning, model explainability, automated validation, and interactive visualization** into a modular analytical platform.

The results demonstrate the value of strong baseline comparisons: Random Forest improves average absolute prediction error, while the Seasonal Naive baseline remains competitive under metrics sensitive to large errors.

The completed system provides a foundation for extending airline demand analysis into automated forecasting and decision-support applications.
