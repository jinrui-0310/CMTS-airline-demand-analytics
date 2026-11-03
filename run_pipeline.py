import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = {
    "data": [
        "src/data_pipeline.py",
        "src/build_duckdb.py",
        "src/build_duckdb_features.py",
        "src/validate_duckdb_features.py",
        "src/build_route_mapping.py",
        "src/export_kpis.py",
        "src/visualize.py",
    ],
    "sql": [
        "src/run_sql_analysis.py",
    ],
    "models": [
        "src/train_elasticnet.py",
        "src/train_xgboost.py",
        "src/train_residual_xgboost.py",
        "src/train_random_forest.py",
    ],
    "analysis": [
        "src/analyze_errors.py",
        "src/walk_forward_validation.py",
        "src/ablation_study.py",
        "src/shap_analysis.py",
        "src/dashboard_data_audit.py",
    ],
}

def run_script(script):
    path = ROOT / script

    if not path.is_file():
        raise FileNotFoundError(path)

    print(f"\n{'=' * 65}", flush=True)
    print(f"RUNNING: {script}", flush=True)
    print(f"{'=' * 65}", flush=True)

    start = time.perf_counter()

    subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        check=True,
    )

    elapsed = time.perf_counter() - start

    print(
        f"COMPLETED: {script} ({elapsed:.1f}s)",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="CMTS Airline Demand Analytics Pipeline"
    )

    parser.add_argument(
        "--data",
        action="store_true",
        help="Run data processing and DuckDB construction",
    )

    parser.add_argument(
        "--sql",
        action="store_true",
        help="Run SQL validation and analysis queries",
    )

    parser.add_argument(
        "--models",
        action="store_true",
        help="Train all models",
    )

    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Run evaluation, SHAP, and dashboard audit",
    )

    args = parser.parse_args()

    selected = [
        name for name in STAGES
        if getattr(args, name)
    ]

    if not selected:
        selected = list(STAGES)

    print("\nCMTS AIRLINE DEMAND ANALYTICS")
    print(f"Selected stages: {', '.join(selected)}")

    pipeline_start = time.perf_counter()

    for stage in selected:
        print(f"\n>>> STAGE: {stage.upper()}", flush=True)

        for script in STAGES[stage]:
            run_script(script)

    elapsed = time.perf_counter() - pipeline_start

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total execution time: {elapsed / 60:.1f} minutes")
    print("=" * 65)


if __name__ == "__main__":
    main()
