from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "airline_analytics.duckdb"

EXPECTED_SOURCE_ROWS = 19_948_408
EXPECTED_ELIGIBLE_ROWS = 8_815_216


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB database not found: {DB_PATH}")

    print("Step 1/4: Connecting to DuckDB...")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        required = ["route_quarter", "route_time_lookup", "route_features"]

        for name in required:
            exists = con.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = ?
                """,
                [name],
            ).fetchone()[0]

            if not exists:
                raise RuntimeError(f"Missing table or view: {name}")

        print("Step 2/4: Checking row counts...")

        source_count = con.execute(
            "SELECT COUNT(*) FROM route_quarter"
        ).fetchone()[0]

        lookup_count = con.execute(
            "SELECT COUNT(*) FROM route_time_lookup"
        ).fetchone()[0]

        feature_count = con.execute(
            "SELECT COUNT(*) FROM route_features"
        ).fetchone()[0]

        eligible_count = con.execute(
            """
            SELECT COUNT(*)
            FROM route_features
            WHERE passengers_lag1 IS NOT NULL
              AND passengers_lag4 IS NOT NULL
            """
        ).fetchone()[0]

        print(f"Source rows:            {source_count:,}")
        print(f"Lookup rows:            {lookup_count:,}")
        print(f"Feature rows:           {feature_count:,}")
        print(f"Eligible modeling rows: {eligible_count:,}")

        assert source_count == EXPECTED_SOURCE_ROWS, "Unexpected source count"
        assert lookup_count == source_count, "Lookup count mismatch"
        assert feature_count == source_count, "Feature count mismatch"
        assert eligible_count == EXPECTED_ELIGIBLE_ROWS, "Eligible count mismatch"

        print("Step 3/4: Checking duplicate keys...")

        duplicates = con.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT
                    OriginCityMarketID,
                    DestCityMarketID,
                    route_id,
                    Year,
                    Quarter
                FROM route_features
                GROUP BY ALL
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        print(f"Duplicate feature keys: {duplicates:,}")
        assert duplicates == 0, "Duplicate feature keys detected"

        print("Step 4/4: Checking lag alignment...")

        invalid_lags = con.execute(
            """
            SELECT COUNT(*)
            FROM route_features f
            LEFT JOIN route_time_lookup l1
              ON f.OriginCityMarketID = l1.OriginCityMarketID
             AND f.DestCityMarketID = l1.DestCityMarketID
             AND f.route_id = l1.route_id
             AND l1.quarter_index = f.quarter_index - 1
            LEFT JOIN route_time_lookup l4
              ON f.OriginCityMarketID = l4.OriginCityMarketID
             AND f.DestCityMarketID = l4.DestCityMarketID
             AND f.route_id = l4.route_id
             AND l4.quarter_index = f.quarter_index - 4
            WHERE f.passengers_lag1 IS DISTINCT FROM l1.passengers
               OR f.passengers_lag4 IS DISTINCT FROM l4.passengers
            """
        ).fetchone()[0]

        print(f"Invalid lag values: {invalid_lags:,}")
        assert invalid_lags == 0, "Lag alignment validation failed"

        print("\nSUCCESS: All DuckDB feature validations passed.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
