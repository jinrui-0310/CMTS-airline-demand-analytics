from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Plot Settings
# ============================================================

plt.rcParams.update({
    "figure.figsize": (12, 5),
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_figure(filename):
    """Save figure and close it."""
    output_path = FIGURE_DIR / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# Figure 1: Passenger Demand Trends
# ============================================================

def plot_demand():

    df = pd.read_csv(TABLE_DIR / "quarterly_demand.csv")

    fig, ax = plt.subplots()

    ax.plot(
        df["period"],
        df["total_route_passengers"],
        linewidth=2
    )

    ax.set_title("Quarterly Airline Passenger Demand (2010–2025)")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("DB1B Sample Passenger Weight")

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x / 1e6:.1f}M")
    )

    # Show only yearly labels
    tick_positions = list(range(0, len(df), 4))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        df["period"].iloc[tick_positions],
        rotation=45,
        ha="right"
    )

    ax.grid(alpha=0.2)

    save_figure("01_passenger_demand.png")


# ============================================================
# Figure 2: Weighted Average Fare
# ============================================================

def plot_fare():

    df = pd.read_csv(TABLE_DIR / "quarterly_fare.csv")

    fig, ax = plt.subplots()

    ax.plot(
        df["period"],
        df["weighted_avg_fare"],
        linewidth=2
    )

    ax.set_title("Passenger-Weighted Average Airfare (2010–2025)")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Average Fare (USD, nominal)")

    tick_positions = list(range(0, len(df), 4))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        df["period"].iloc[tick_positions],
        rotation=45,
        ha="right"
    )

    ax.grid(alpha=0.2)

    save_figure("02_average_fare.png")


# ============================================================
# Figure 3: Market Concentration (HHI)
# ============================================================

def plot_hhi():

    df = pd.read_csv(TABLE_DIR / "quarterly_hhi.csv")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["period"],
        df["avg_market_hhi"],
        linewidth=2.5,
        marker="o",
        markersize=3
    )

    ax.set_title(
        "Average Airline Market Concentration (2010–2025)"
    )

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Average HHI (0–1 scale)")

    # Automatically zoom into the actual HHI range
    hhi_min = df["avg_market_hhi"].min()
    hhi_max = df["avg_market_hhi"].max()

    padding = (hhi_max - hhi_min) * 0.15

    ax.set_ylim(
        hhi_min - padding,
        hhi_max + padding
    )

    tick_positions = list(range(0, len(df), 4))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        df["period"].iloc[tick_positions],
        rotation=45,
        ha="right"
    )

    ax.grid(alpha=0.2)

    # Explain the zoomed axis
    ax.text(
        0.01,
        0.02,
        "Note: Y-axis zoomed to highlight changes in HHI.",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.7
    )

    save_figure("03_market_concentration.png")


# ============================================================
# Figure 4: Top 20 O&D Markets
# ============================================================

def plot_top_markets():

    df = pd.read_csv(TABLE_DIR / "top20_markets_2024.csv")

    df["market"] = (
        df["OriginCityMarketID"].astype(str)
        + " → "
        + df["DestCityMarketID"].astype(str)
    )

    # Reverse order for horizontal bar chart
    df = df.sort_values("total_passengers")

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.barh(
        df["market"],
        df["total_passengers"]
    )

    ax.set_title("Top 20 O&D Markets by Passenger Demand (2024)")
    ax.set_xlabel("DB1B Sample Passenger Weight")
    ax.set_ylabel("Directional O&D Market")

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x / 1e6:.1f}M")
    )

    ax.grid(axis="x", alpha=0.2)

    save_figure("04_top20_markets.png")


# ============================================================
# Main
# ============================================================

def main():

    print("CMTS: Generating KPI Visualizations")

    plot_demand()
    plot_fare()
    plot_hhi()
    plot_top_markets()

    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()