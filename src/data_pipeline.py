from pathlib import Path
import pandas as pd
import json


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROUTE_PATH = PROJECT_ROOT / "data" / "raw" / "DB1B_RoutePanel_Core_2010Q1_2025Q2.csv"

HHI_PATH = PROJECT_ROOT / "data" / "raw" / "HHI_market_quarter_2010Q1_2025Q2.csv"



OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "tables"

OUTPUT_PATH = OUTPUT_DIR / "cmts_route_quarter.csv"
REPORT_PATH = REPORT_DIR / "data_quality_report.json"

CHUNK_SIZE = 200_000

MERGE_KEYS = [
    "Year",
    "Quarter",
    "OriginCityMarketID",
    "DestCityMarketID",
]

ROUTE_COLUMNS = [
    "Year",
    "Quarter",
    "OriginCityMarketID",
    "DestCityMarketID",
    "Origin",
    "Dest",
    "route_id",
    "passengers",
    "avg_fare",
    "route_distance",
    "nonstop_distance",
    "n_records",
    "market_id",
    "time_id",
    "route_panel_id",
    "market_passengers",
    "route_share",
]

HHI_COLUMNS = [
    "Year",
    "Quarter",
    "OriginCityMarketID",
    "DestCityMarketID",
    "HHI_TK",
    "n_carriers_TK",
]


# ============================================================
# 2. VALIDATION
# ============================================================

def validate_file(path, expected_columns, dataset_name):
    """Check file existence and required columns."""

    print(f"\nChecking {dataset_name}...")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    header = pd.read_csv(path, nrows=0)

    missing = set(expected_columns) - set(header.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing}"
        )

    size_gb = path.stat().st_size / (1024 ** 3)

    print(f"File: {path.name}")
    print(f"Size: {size_gb:.3f} GB")
    print(f"Columns: {len(header.columns)}")
    print("Column validation passed.")


def load_hhi():
    """Load HHI panel and validate unique market-quarter keys."""

    print("\nLoading HHI panel...")

    hhi = pd.read_csv(HHI_PATH)

    duplicate_count = int(
        hhi.duplicated(subset=MERGE_KEYS).sum()
    )

    print(f"HHI rows: {len(hhi):,}")
    print(f"Duplicate merge keys: {duplicate_count:,}")

    if duplicate_count > 0:
        raise ValueError(
            "HHI panel has duplicate market-quarter keys."
        )

    missing_hhi = int(hhi["HHI_TK"].isna().sum())
    missing_carriers = int(hhi["n_carriers_TK"].isna().sum())

    print(f"Missing HHI: {missing_hhi:,}")
    print(f"Missing carrier count: {missing_carriers:,}")

    return hhi


# ============================================================
# 3. CHUNKED MERGE
# ============================================================

def merge_route_hhi(hhi):
    """Merge large route panel with HHI using chunked processing."""

    print("\nStarting chunked merge...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    matched_rows = 0
    missing_hhi_rows = 0
    missing_carrier_rows = 0
    chunk_count = 0

    for chunk in pd.read_csv(
        ROUTE_PATH,
        chunksize=CHUNK_SIZE,
    ):
        chunk_count += 1

        merged = chunk.merge(
            hhi,
            on=MERGE_KEYS,
            how="left",
            validate="many_to_one",
            indicator=True,
        )

        total_rows += len(merged)

        matched_rows += int(
            merged["_merge"].eq("both").sum()
        )

        missing_hhi_rows += int(
            merged["HHI_TK"].isna().sum()
        )

        missing_carrier_rows += int(
            merged["n_carriers_TK"].isna().sum()
        )

        merged = merged.drop(columns=["_merge"])

        merged.to_csv(
            OUTPUT_PATH,
            mode="w" if chunk_count == 1 else "a",
            header=(chunk_count == 1),
            index=False,
        )

        print(
            f"Chunk {chunk_count}: "
            f"{len(chunk):,} rows processed | "
            f"Total: {total_rows:,}"
        )

    report = {
        "route_rows": total_rows,
        "hhi_rows": len(hhi),
        "chunks_processed": chunk_count,
        "matched_rows": matched_rows,
        "unmatched_rows": total_rows - matched_rows,
        "merge_match_rate": (
            matched_rows / total_rows if total_rows else None
        ),
        "missing_hhi_rows": missing_hhi_rows,
        "hhi_availability_rate": (
            (total_rows - missing_hhi_rows) / total_rows
            if total_rows else None
        ),
        "missing_carrier_rows": missing_carrier_rows,
        "output_file": str(OUTPUT_PATH),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    return report


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("CMTS Airline Demand Analytics")
    print("Day 1: Route-Level Panel + HHI Integration")
    print("=" * 60)

    validate_file(
        ROUTE_PATH,
        ROUTE_COLUMNS,
        "Route-Level Panel",
    )

    validate_file(
        HHI_PATH,
        HHI_COLUMNS,
        "HHI Panel",
    )

    hhi = load_hhi()

    report = merge_route_hhi(hhi)

    print("\n" + "=" * 60)
    print("MERGE COMPLETED")
    print("=" * 60)

    print(f"Total rows: {report['route_rows']:,}")
    print(f"Matched rows: {report['matched_rows']:,}")
    print(f"Unmatched rows: {report['unmatched_rows']:,}")

    print(
        f"Merge match rate: "
        f"{report['merge_match_rate']:.2%}"
    )

    print(
        f"HHI availability rate: "
        f"{report['hhi_availability_rate']:.2%}"
    )

    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Quality report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

