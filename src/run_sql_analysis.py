from pathlib import Path
import time

import duckdb


ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "data/processed/airline_analytics.duckdb"

SQL_DIR = ROOT / "sql"

OUTPUT_DIR = ROOT / "outputs/tables/sql_analysis"

SQL_FILES = [
    "01_data_validation.sql",
    "02_market_analysis.sql",
    "03_route_features.sql",
    "06_modeling_dataset.sql",
]


def split_statements(sql):
    """
    Split simple SQL scripts into statements.

    These project SQL files contain SELECT statements
    separated by semicolons.
    """
    return [
        statement.strip()
        for statement in sql.split(";")
        if statement.strip()
    ]


def main():
    start = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )

    try:
        for filename in SQL_FILES:
            path = SQL_DIR / filename

            print(
                f"\nRunning SQL file: {filename}",
                flush=True,
            )

            sql = path.read_text(encoding="utf-8")

            statements = split_statements(sql)

            prefix = path.stem

            for index, statement in enumerate(
                statements,
                start=1,
            ):
                # Remove leading SQL comments.
                lines = statement.splitlines()

                cleaned = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("--")
                ).strip()

                if not cleaned:
                    continue

                df = con.execute(cleaned).df()

                output_path = (
                    OUTPUT_DIR
                    / f"{prefix}_query_{index:02d}.csv"
                )

                df.to_csv(output_path, index=False)

                print(
                    f"Query {index}: "
                    f"{len(df):,} rows -> "
                    f"{output_path.name}",
                    flush=True,
                )

        print(
            f"\nSQL analysis completed in "
            f"{time.time() - start:.1f}s"
        )

    finally:
        con.close()


if __name__ == "__main__":
    main()
