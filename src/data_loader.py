"""Shared DuckDB data loader for airline demand models."""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "airline_analytics.duckdb"

BASE_COLUMNS = [
    "Year",
    "Quarter",
    "OriginCityMarketID",
    "DestCityMarketID",
    "route_id",
    "target_passengers",
    "passengers_lag1",
    "passengers_lag4",
    "fare_lag1",
    "hhi_lag1",
    "carriers_lag1",
    "route_distance",
    "nonstop_distance",
]

ORDER_COLUMNS = [
    "Year",
    "Quarter",
    "OriginCityMarketID",
    "DestCityMarketID",
    "route_id",
]


def load_modeling_data():
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"DuckDB database not found: {DB_PATH}")

    columns = ", ".join(BASE_COLUMNS)
    order = ", ".join(ORDER_COLUMNS)

    query = f"""
        SELECT {columns}
        FROM route_features
        WHERE passengers_lag1 IS NOT NULL
          AND passengers_lag4 IS NOT NULL
          AND target_passengers IS NOT NULL
          AND (Year < 2025 OR (Year = 2025 AND Quarter <= 2))
        ORDER BY {order}
    """

    con = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        return con.execute(query).df()
    finally:
        con.close()