from pathlib import Path
import time

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data/processed/airline_analytics.duckdb"
SQL_DIR = ROOT / "sql"


def run_sql(con, filename):
    path = SQL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    print(f"Executing SQL: {filename}", flush=True)

    sql = path.read_text(encoding="utf-8")

    con.execute(sql)


def main():
    start = time.time()

    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    con = duckdb.connect(str(DB_PATH))

    try:
        print(
            "Step 1/4: Validating source keys...",
            flush=True,
        )

        source_count = con.execute(
            "SELECT COUNT(*) FROM route_quarter"
        ).fetchone()[0]

        duplicate_count = con.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT
                    OriginCityMarketID,
                    DestCityMarketID,
                    route_id,
                    Year,
                    Quarter
                FROM route_quarter
                GROUP BY 1, 2, 3, 4, 5
                HAVING COUNT(*) > 1
            ) t
        """).fetchone()[0]

        if duplicate_count:
            raise ValueError(
                f"Found {duplicate_count:,} duplicate route-quarter keys"
            )

        print(
            f"Source rows: {source_count:,}",
            flush=True,
        )

        print(
            "Step 2/4: Creating lookup table...",
            flush=True,
        )

        run_sql(
            con,
            "04_route_feature_engineering.sql",
        )

        print(
            "Step 3/4: Creating route features...",
            flush=True,
        )

        run_sql(
            con,
            "05_build_route_features.sql",
        )

        print(
            "Step 4/4: Validating...",
            flush=True,
        )

        feature_count = con.execute(
            "SELECT COUNT(*) FROM route_features"
        ).fetchone()[0]

        eligible_count = con.execute("""
            SELECT COUNT(*)
            FROM route_features
            WHERE passengers_lag1 IS NOT NULL
              AND passengers_lag4 IS NOT NULL
        """).fetchone()[0]

        if feature_count != source_count:
            raise ValueError(
                f"Row count mismatch: "
                f"{feature_count:,} vs {source_count:,}"
            )

        print(f"Feature rows: {feature_count:,}")
        print(
            f"Eligible modeling rows: {eligible_count:,}"
        )

        print(
            f"Completed in {time.time() - start:.1f}s"
        )

    finally:
        con.close()


if __name__ == "__main__":
    main()
