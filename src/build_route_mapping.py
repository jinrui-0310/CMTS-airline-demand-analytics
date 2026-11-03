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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("Building route mapping...", flush=True)

    conn = duckdb.connect(str(DB_PATH), read_only=True)

    query = """
        SELECT
            OriginCityMarketID,
            DestCityMarketID,
            route_id,
            Origin,
            Dest,
            COUNT(*) AS observations
        FROM route_quarter
        GROUP BY
            OriginCityMarketID,
            DestCityMarketID,
            route_id,
            Origin,
            Dest
    """

    try:
        df = conn.execute(query).df()
    finally:
        conn.close()

    output_path = OUTPUT_DIR / "route_mapping.csv"

    df.to_csv(output_path, index=False)

    print(f"Mapping rows: {len(df):,}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()