from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

CSV_PATH = DATA_DIR / "cmts_route_quarter.csv"
PARQUET_PATH = DATA_DIR / "route_quarter.parquet"
DB_PATH = DATA_DIR / "airline_analytics.duckdb"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found: {CSV_PATH}")

    print("Step 1/3: Converting CSV to Parquet...")

    con = duckdb.connect(str(DB_PATH))

    try:
        # DuckDB reads the CSV and writes compressed Parquet.
        # No need to load the entire dataset into pandas.
        con.execute(
            """
            COPY (
                SELECT *
                FROM read_csv_auto(
                    ?,
                    header = true,
                    sample_size = -1
                )
            )
            TO ?
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [str(CSV_PATH), str(PARQUET_PATH)],
        )

        print("Step 2/3: Creating DuckDB view...")

        # Store the file path in the view definition.
        parquet_sql_path = "data/processed/route_quarter.parquet"

        con.execute(
            f"""
            CREATE OR REPLACE VIEW route_quarter AS
            SELECT *
            FROM read_parquet('{parquet_sql_path}')
            """
        )

        print("Step 3/3: Validating row count...")

        count = con.execute(
            "SELECT COUNT(*) FROM route_quarter"
        ).fetchone()[0]

        print(f"Rows: {count:,}")
        print(f"Parquet: {PARQUET_PATH}")
        print(f"DuckDB: {DB_PATH}")
        print("Migration completed.")

    finally:
        con.close()


if __name__ == "__main__":
    main()