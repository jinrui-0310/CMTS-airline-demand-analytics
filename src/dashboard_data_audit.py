import duckdb
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "airline_analytics.duckdb"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "random_forest_predictions.csv"
)

MAPPING_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "route_mapping.csv"
)


def main():

    print("\n========== DASHBOARD DATA AUDIT ==========")

    conn = duckdb.connect(str(DB_PATH), read_only=True)

    # --------------------------------------------------------
    # 1. HHI consistency
    # --------------------------------------------------------

    print("\n[1] Checking HHI consistency...")

    query = """
        SELECT
            Year,
            Quarter,
            OriginCityMarketID,
            DestCityMarketID,

            COUNT(DISTINCT HHI_TK) AS distinct_hhi,
            COUNT(DISTINCT n_carriers_TK) AS distinct_carriers

        FROM route_quarter

        GROUP BY
            Year,
            Quarter,
            OriginCityMarketID,
            DestCityMarketID

        HAVING
            COUNT(DISTINCT HHI_TK) > 1
            OR COUNT(DISTINCT n_carriers_TK) > 1
    """

    hhi_issues = conn.execute(query).df()

    print(
        f"Market-quarter groups with inconsistent HHI "
        f"or carrier counts: {len(hhi_issues):,}"
    )

    if not hhi_issues.empty:
        print(hhi_issues.head(10).to_string(index=False))

    conn.close()

    # --------------------------------------------------------
    # 2. Prediction uniqueness
    # --------------------------------------------------------

    print("\n[2] Checking prediction uniqueness...")

    predictions = pd.read_csv(PREDICTIONS_PATH)

    prediction_keys = [
        "Year",
        "Quarter",
        "OriginCityMarketID",
        "DestCityMarketID",
        "route_id",
    ]

    duplicates = predictions.duplicated(
        prediction_keys,
        keep=False,
    )

    print(
        f"Duplicate prediction rows: "
        f"{duplicates.sum():,}"
    )

    if duplicates.any():
        print(
            predictions.loc[
                duplicates,
                prediction_keys
            ].head(10).to_string(index=False)
        )

    # --------------------------------------------------------
    # 3. Mapping uniqueness
    # --------------------------------------------------------

    print("\n[3] Checking airport mapping...")

    mapping = pd.read_csv(MAPPING_PATH)

    mapping_keys = [
        "OriginCityMarketID",
        "DestCityMarketID",
        "route_id",
    ]

    mapping_counts = (
        mapping.groupby(mapping_keys)
        .size()
        .reset_index(name="mapping_count")
    )

    multiple_mappings = mapping_counts[
        mapping_counts["mapping_count"] > 1
    ]

    print(
        f"Route keys with multiple airport mappings: "
        f"{len(multiple_mappings):,}"
    )

    if not multiple_mappings.empty:
        print(
            multiple_mappings.head(10).to_string(index=False)
        )

    # --------------------------------------------------------
    # 4. Prediction-to-mapping coverage
    # --------------------------------------------------------

    print("\n[4] Checking mapping coverage...")

    available_keys = mapping[
        mapping_keys
    ].drop_duplicates()

    matched = predictions[
        mapping_keys
    ].drop_duplicates().merge(
        available_keys,
        on=mapping_keys,
        how="left",
        indicator=True,
    )

    unmatched = matched[
        matched["_merge"] == "left_only"
    ]

    print(
        f"Forecast route keys without airport mapping: "
        f"{len(unmatched):,}"
    )

    # --------------------------------------------------------
    # 5. Forecast coverage
    # --------------------------------------------------------

    print("\n[5] Checking forecast coverage...")

    coverage = (
        predictions.groupby(
            ["Year", "Quarter"]
        )
        .size()
        .reset_index(name="observations")
    )

    print(coverage.to_string(index=False))

    print("\n========== AUDIT COMPLETED ==========")


if __name__ == "__main__":
    main()
