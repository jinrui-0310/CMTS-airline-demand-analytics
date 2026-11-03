# Airline Demand Analytics

A Python project for analyzing U.S. airline route-level demand and forecasting quarterly passenger volumes.

The project uses DuckDB and SQL to process airline market data, builds time-aware features for forecasting, compares several machine learning models, and provides a Streamlit dashboard for exploring the results.

The dataset covers 2010 Q1 through 2025 Q2, with approximately 20 million route-quarter observations.

## Project Structure

```text
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
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## Data Pipeline

The pipeline consists of four stages:

| Stage | Description |
|---|---|
| Data | Integrates raw datasets, builds the DuckDB database, generates features, and exports analytical tables |
| SQL | Runs data validation and market analysis queries |
| Models | Trains ElasticNet, XGBoost, Residual XGBoost, and Random Forest |
| Analysis | Performs error analysis, walk-forward validation, ablation studies, SHAP analysis, and dashboard data auditing |

The complete workflow is managed by `run_pipeline.py`.

```text
Raw CSV Files
      |
      v
Data Cleaning and Integration
      |
      v
DuckDB + Parquet
      |
      v
SQL Feature Engineering
      |
      v
Forecasting Models
      |
      v
Model Evaluation
      |
      v
Streamlit Dashboard
```

### Data Processing

The project combines route-level airline observations with market concentration data.

The processed dataset contains:

| Dataset | Observations |
|---|---:|
| Integrated route-quarter records | 19,948,408 |
| Eligible modeling records | 8,815,216 |
| Test predictions | 992,290 |

Feature validation checks duplicate route-quarter keys and the temporal alignment of lagged variables.

The latest validation run found no duplicate keys or invalid lag-1/lag-4 alignments.

### SQL Analysis

SQL scripts are organized by purpose:

| File | Purpose |
|---|---|
| `01_data_validation.sql` | Data validation |
| `02_market_analysis.sql` | Market analysis |
| `03_route_features.sql` | Route-level feature analysis |
| `04_route_feature_engineering.sql` | Feature engineering |
| `05_build_route_features.sql` | Feature table construction |
| `06_modeling_dataset.sql` | Modeling dataset analysis |

DuckDB is used for analytical queries and feature generation. Parquet provides storage for the processed route-quarter data.

## Forecasting Models

Five approaches are evaluated:

- Seasonal Naive
- ElasticNet
- XGBoost
- Residual XGBoost
- Random Forest

The evaluation period is 2024 Q1 through 2025 Q2.

### Test Set Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Seasonal Naive | 16.99 | **86.41** | **0.9831** |
| ElasticNet | 30.77 | 166.44 | 0.9373 |
| XGBoost | 17.78 | 130.38 | 0.9615 |
| Residual XGBoost | 16.67 | 101.64 | 0.9766 |
| Random Forest | **16.54** | 96.34 | 0.9790 |

Random Forest achieves the lowest MAE, while the seasonal naive baseline performs best on RMSE and R².

The results suggest that more complex models do not necessarily outperform a strong historical baseline across all evaluation metrics.

Training sample sizes differ between models, so the comparison reflects the current implementations rather than a fully controlled benchmark.

## Model Analysis

The project includes several additional analyses.

**Error Analysis**

Examines prediction errors and identifies cases where models perform poorly.

**Walk-Forward Validation**

Evaluates model performance across sequential time periods.

**Ablation Study**

Measures how different feature groups affect forecasting performance.

**SHAP Analysis**

Examines feature contributions to model predictions.

**Dashboard Data Audit**

Validates the consistency of data used by the Streamlit dashboard.

## Dashboard

The Streamlit dashboard provides an interface for exploring the generated analytical results.

Run it from the project root:

```bash
streamlit run app.py/app.py
```

The dashboard requires locally generated data files. These are not included in the Git repository.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd CMTS-airline-demand-analytics
```

### 2. Set up Python

Python 3.11 was used during development.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The development environment uses DuckDB 2.0.0.

### 4. Prepare the data

Place the following source files in `data/raw/`:

```text
data/raw/
├── DB1B_RoutePanel_Core_2010Q1_2025Q2.csv
└── HHI_market_quarter_2010Q1_2025Q2.csv
```

The source datasets and generated database files are excluded from Git due to their size.

A public download location for these exact input files is not currently provided in this repository.

## Running the Pipeline

To execute the full workflow:

```bash
python run_pipeline.py
```

The pipeline runs the following stages in order:

```text
data → sql → models → analysis
```

Individual stages can be executed separately:

```bash
python run_pipeline.py --data
python run_pipeline.py --sql
python run_pipeline.py --models
python run_pipeline.py --analysis
```

Multiple stages can also be selected:

```bash
python run_pipeline.py --data --sql
```

The stages have dependencies. Data processing must be completed before model training, and model training must be completed before downstream model analysis.

## Reproducibility Notes

- Raw data files are not included in this repository.
- Generated DuckDB databases, Parquet files, model artifacts, and analysis outputs are excluded from Git.
- The reported metrics come from a completed local pipeline run.
- The Python dependencies are not fully version-pinned.
- The development environment uses Python 3.11 and DuckDB 2.0.0.
- Some DuckDB views reference local Parquet files, so commands should be run from the project root.
- Model training configurations and training sample sizes differ across approaches.

