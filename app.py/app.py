import duckdb
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="US Airline Demand Analytics", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "airline_analytics.duckdb"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
BLUE, ORANGE, DARK = "#2563EB", "#F59E0B", "#0F172A"

st.markdown("""<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1450px;}
[data-testid="stMetric"] {background: #f3f7fc; padding: 14px 18px; border-radius: 12px; border: 1px solid #e2e8f0;}
[data-testid="stMetricLabel"] {font-weight: 600;}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def run_query(query, params=()):
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        return conn.execute(query, params).df()

@st.cache_data(show_spinner=False)
def load_market_overview():
    return run_query("""
        SELECT Year, Quarter, SUM(passengers) AS total_passengers,
               SUM(passengers * avg_fare) / NULLIF(SUM(CASE WHEN avg_fare IS NOT NULL THEN passengers ELSE 0 END), 0) AS weighted_avg_fare,
               COUNT(*) AS route_records
        FROM route_quarter GROUP BY Year, Quarter ORDER BY Year, Quarter
    """)

@st.cache_data(show_spinner=False)
def load_top_airport_pairs(start_year, end_year, limit=10):
    # Airport-pair totals include every itinerary with these endpoints.
    return run_query("""
        SELECT Origin, Dest,
               SUM(passengers) AS passengers,
               SUM(CASE WHEN avg_fare IS NOT NULL THEN passengers * avg_fare ELSE 0 END)
                 / NULLIF(SUM(CASE WHEN avg_fare IS NOT NULL THEN passengers ELSE 0 END), 0) AS avg_fare,
               COUNT(DISTINCT Year * 10 + Quarter) AS quarters
        FROM route_quarter
        WHERE Year BETWEEN ? AND ? AND Origin IS NOT NULL AND Dest IS NOT NULL
        GROUP BY Origin, Dest
        ORDER BY passengers DESC
        LIMIT ?
    """, (start_year, end_year, limit))


@st.cache_data(show_spinner=False)
def load_airports():
    return run_query("""
        SELECT Origin AS airport FROM route_quarter WHERE Origin IS NOT NULL
        UNION SELECT Dest AS airport FROM route_quarter WHERE Dest IS NOT NULL
        ORDER BY airport
    """)["airport"].tolist()

@st.cache_data(show_spinner=False)
def load_destinations(origin):
    return run_query("SELECT DISTINCT Dest FROM route_quarter WHERE Origin = ? ORDER BY Dest", (origin,))["Dest"].tolist()

@st.cache_data(show_spinner=False)
def load_route_history(origin, destination):
    return run_query("""
        SELECT Year, Quarter, SUM(passengers) AS passengers,
               SUM(passengers * avg_fare) / NULLIF(SUM(CASE WHEN avg_fare IS NOT NULL THEN passengers ELSE 0 END), 0) AS avg_fare,
               MIN(HHI_TK) AS hhi, MIN(n_carriers_TK) AS n_carriers,
               MIN(nonstop_distance) AS nonstop_distance,
               COUNT(DISTINCT CAST(OriginCityMarketID AS TEXT) || '-' || CAST(DestCityMarketID AS TEXT)) AS market_count
        FROM route_quarter WHERE Origin = ? AND Dest = ?
        GROUP BY Year, Quarter ORDER BY Year, Quarter
    """, (origin, destination))

@st.cache_data(show_spinner=False)
def load_predictions():
    return pd.read_csv(TABLE_DIR / "random_forest_predictions.csv")

@st.cache_data(show_spinner=False)
def load_route_mapping():
    return pd.read_csv(TABLE_DIR / "route_mapping.csv")

@st.cache_data(show_spinner=False)
def load_results_csv(filename):
    path = TABLE_DIR / filename
    return pd.read_csv(path) if path.exists() else None

def display_saved_figure(filename, caption=None):
    path = FIGURE_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not found: {filename}")

def add_period(df):
    result = df.copy()
    result["Period"] = result["Year"].astype(str) + " Q" + result["Quarter"].astype(str)
    result["Date"] = pd.to_datetime(result["Year"].astype(str) + "-" + ((result["Quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01")
    return result.sort_values(["Year", "Quarter"])

def polish(fig, height=420, x_title="Quarter", y_title=None):
    fig.update_layout(template="plotly_white", height=height, hovermode="x unified",
                      margin=dict(l=12, r=12, t=38, b=20),
                      xaxis_title=x_title, yaxis_title=y_title,
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig

st.sidebar.markdown("# ✈️ Airline Analytics")
st.sidebar.caption("U.S. Airline Market & Passenger Demand Analytics")
page = st.sidebar.radio("Navigation", ["Market Overview", "Route Explorer", "Demand Forecasting", "Model Insights"])
st.sidebar.divider()
st.sidebar.markdown("**CMTS Capstone Project**")
st.sidebar.caption("DB1B-based airline market analysis · 2010 Q1–2025 Q2")
st.sidebar.caption("Python · DuckDB · Parquet · scikit-learn · SHAP · Streamlit")

if page == "Market Overview":
    st.title("U.S. Airline Market Overview")
    st.write("Explore DB1B itinerary-based passenger volume, fares, leading airport pairs, and pandemic recovery.")
    with st.spinner("Loading quarterly market data..."):
        all_df = load_market_overview()
    if all_df.empty:
        st.warning("No market data found.")
        st.stop()
    all_df = add_period(all_df)
    min_year, max_year = int(all_df.Year.min()), int(all_df.Year.max())
    years = st.slider("Year range", min_year, max_year, (min_year, max_year), key="market_years")
    df = all_df[all_df.Year.between(*years)].copy()
    latest = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected-period DB1B passenger volume", f"{df.total_passengers.sum():,.0f}")
    c2.metric("Latest weighted average fare", f"${latest.weighted_avg_fare:,.2f}" if pd.notna(latest.weighted_avg_fare) else "N/A")
    c3.metric("Quarters shown", str(len(df)))
    st.caption("Volume represents the processed DB1B records, not a census of all U.S. air travelers. Quarterly fare is passenger-weighted over records with a reported fare.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Quarterly Passenger Demand")
        fig = px.line(df, x="Date", y="total_passengers", markers=True, color_discrete_sequence=[BLUE])
        st.plotly_chart(polish(fig, 390, y_title="DB1B passenger volume"), use_container_width=True)
    with c2:
        st.subheader("Quarterly Average Airfare")
        fig = px.line(df, x="Date", y="weighted_avg_fare", markers=True, color_discrete_sequence=[ORANGE])
        st.plotly_chart(polish(fig, 390, y_title="Weighted average fare ($)"), use_container_width=True)

    st.divider()
    st.header("Top Airport-Pair Markets")
    st.caption("Ranked by summed DB1B itinerary passengers sharing the same origin and destination airports; both nonstop and connecting itineraries are included. Direction is preserved (A → B differs from B → A).")
    with st.spinner("Ranking airport pairs..."):
        top_pairs = load_top_airport_pairs(*years)
    if top_pairs.empty:
        st.info("No airport pairs found for this year range.")
    else:
        top_pairs["Airport Pair"] = top_pairs["Origin"] + " → " + top_pairs["Dest"]
        chart = px.bar(top_pairs.sort_values("passengers"), x="passengers", y="Airport Pair",
                       orientation="h", color_discrete_sequence=[BLUE],
                       hover_data={"avg_fare": ":.2f", "quarters": True})
        st.plotly_chart(polish(chart, 430, x_title="DB1B passenger volume", y_title=None),
                        use_container_width=True)
        with st.expander("Top airport-pair data"):
            st.dataframe(top_pairs[["Airport Pair", "passengers", "avg_fare", "quarters"]],
                         hide_index=True, use_container_width=True)
            st.download_button("Download top airport pairs CSV",
                               top_pairs.to_csv(index=False), "top_airport_pairs.csv", "text/csv")

    st.divider()
    st.header("COVID-19 Shock & Recovery")
    st.caption("Quarter-matched comparison with 2019 as the pre-pandemic reference. The 2025 dataset contains only Q1–Q2; missing quarters are never treated as zero.")
    available = all_df[all_df.Year.between(2019, 2025)].copy()
    baseline = available.loc[available.Year == 2019, ["Quarter", "total_passengers", "weighted_avg_fare"]].rename(
        columns={"total_passengers": "base_passengers", "weighted_avg_fare": "base_fare"})
    recovery = available.merge(baseline, on="Quarter", how="left")
    recovery["Passenger change vs 2019 (%)"] = np.where(
        recovery["base_passengers"] > 0,
        (recovery["total_passengers"] / recovery["base_passengers"] - 1) * 100, np.nan)
    recovery["Fare change vs 2019 (%)"] = np.where(
        recovery["base_fare"] > 0,
        (recovery["weighted_avg_fare"] / recovery["base_fare"] - 1) * 100, np.nan)
    if recovery.empty or baseline.empty:
        st.info("2019 baseline data is unavailable.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Passenger Volume vs 2019")
            fig = px.line(recovery, x="Date", y="Passenger change vs 2019 (%)",
                          markers=True, color_discrete_sequence=[BLUE])
            fig.add_hline(y=0, line_dash="dash", line_color="#64748B")
            st.plotly_chart(polish(fig, 390, y_title="Change vs same quarter 2019 (%)"),
                            use_container_width=True)
        with c2:
            st.subheader("Airfare vs 2019")
            fig = px.line(recovery, x="Date", y="Fare change vs 2019 (%)",
                          markers=True, color_discrete_sequence=[ORANGE])
            fig.add_hline(y=0, line_dash="dash", line_color="#64748B")
            st.plotly_chart(polish(fig, 390, y_title="Change vs same quarter 2019 (%)"),
                            use_container_width=True)
        st.caption("Fare changes are nominal, not inflation-adjusted. The baseline is the corresponding quarter in 2019, rather than the full-year 2019 average.")
        with st.expander("COVID/recovery data and export"):
            st.dataframe(recovery.drop(columns=["Date"]), use_container_width=True, hide_index=True)
            st.download_button("Download COVID recovery CSV",
                               recovery.drop(columns=["Date"]).to_csv(index=False),
                               "covid_recovery.csv", "text/csv")

    with st.expander("View quarterly market data"):
        st.dataframe(df.drop(columns="Date"), use_container_width=True, hide_index=True)
        st.download_button("Download market overview CSV",
                           df.drop(columns="Date").to_csv(index=False),
                           "market_overview.csv", "text/csv")

elif page == "Route Explorer":
    st.title("Route Explorer")
    st.write("Analyze airport-pair passenger volume, airfare, and city-market competition indicators.")
    airports = load_airports()
    c1, c2 = st.columns(2)
    with c1:
        origin = st.selectbox("Origin Airport", airports, index=airports.index("LAX") if "LAX" in airports else 0, key="explorer_origin")
    destinations = load_destinations(origin)
    if not destinations:
        st.warning("No destinations available.")
        st.stop()
    with c2:
        destination = st.selectbox("Destination Airport", destinations, index=destinations.index("JFK") if "JFK" in destinations else 0, key="explorer_dest")
    with st.spinner("Loading route history..."):
        df = load_route_history(origin, destination)
    if df.empty:
        st.warning("No route data available.")
        st.stop()
    df = add_period(df)
    affected = df.market_count > 1
    if affected.any():
        st.warning("Multiple city-market combinations occur in some quarters. HHI and carrier counts are hidden for those quarters.")
        df.loc[affected, ["hhi", "n_carriers"]] = np.nan
    st.subheader(f"{origin} → {destination}")
    years = st.slider("Year range", int(df.Year.min()), int(df.Year.max()), (int(df.Year.min()), int(df.Year.max())), key="explorer_years") if df.Year.min() < df.Year.max() else (int(df.Year.min()), int(df.Year.max()))
    df = df[df.Year.between(*years)].copy()
    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest quarterly passenger volume", f"{latest.passengers:,.0f}")
    c2.metric("Latest average fare", f"${latest.avg_fare:,.2f}" if pd.notna(latest.avg_fare) else "N/A")
    c3.metric("Latest city-market HHI", f"{latest.hhi:.3f}" if pd.notna(latest.hhi) else "N/A")
    c4.metric("Nonstop distance", f"{latest.nonstop_distance:,.0f} mi" if pd.notna(latest.nonstop_distance) else "N/A")
    st.caption("HHI and carrier count are city-market measures; they are not independently calculated for an individual airport pair.")
    st.subheader("Passenger Demand")
    fig = px.line(df, x="Date", y="passengers", markers=True, color_discrete_sequence=[BLUE])
    st.plotly_chart(polish(fig, 420, y_title="DB1B passenger volume"), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average Fare")
        fig = px.line(df, x="Date", y="avg_fare", markers=True, color_discrete_sequence=[ORANGE])
        st.plotly_chart(polish(fig, 350, y_title="Average fare ($)"), use_container_width=True)
    with c2:
        st.subheader("City-Market Concentration (HHI)")
        fig = px.line(df, x="Date", y="hhi", markers=True, color_discrete_sequence=[DARK])
        st.plotly_chart(polish(fig, 350, y_title="HHI"), use_container_width=True)
    st.subheader("Number of Carriers (City Market)")
    fig = px.line(df, x="Date", y="n_carriers", markers=True, color_discrete_sequence=[BLUE])
    st.plotly_chart(polish(fig, 360, y_title="Carriers"), use_container_width=True)
    with st.expander("View route data"):
        st.dataframe(df.drop(columns="Date"), use_container_width=True, hide_index=True)
        st.download_button("Download route history CSV", df.drop(columns="Date").to_csv(index=False), f"{origin}_{destination}_history.csv", "text/csv")

elif page == "Demand Forecasting":
    st.title("Route-Level Passenger Demand Forecasting")
    st.write("Explore nonstop and connecting itineraries, or compare their airport-pair totals.")
    st.info("**Historical out-of-sample backtest (2024 Q1–2025 Q2)** — not a forecast of future quarters.")

    with st.spinner("Loading forecast results..."):
        predictions = load_predictions()
        mapping = load_route_mapping()
    if predictions.empty or mapping.empty:
        st.warning("Prediction or airport mapping data is missing.")
        st.stop()

    keys = ["OriginCityMarketID", "DestCityMarketID", "route_id"]
    airport_routes = mapping[["Origin", "Dest", *keys]].drop_duplicates()
    airport_routes = airport_routes.merge(predictions[keys].drop_duplicates(), on=keys, how="inner")
    origins = sorted(airport_routes["Origin"].dropna().unique().tolist())
    if not origins:
        st.warning("No airport pairs with test predictions found.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        origin = st.selectbox("Origin Airport", origins,
                              index=origins.index("LAX") if "LAX" in origins else 0,
                              key="forecast_origin")
    origin_routes = airport_routes.loc[airport_routes["Origin"] == origin]
    destinations = sorted(origin_routes["Dest"].dropna().unique().tolist())
    with c2:
        destination = st.selectbox("Destination Airport", destinations,
                                   index=destinations.index("JFK") if "JFK" in destinations else 0,
                                   key="forecast_destination")

    selected_routes = origin_routes.loc[origin_routes["Dest"] == destination, keys].drop_duplicates()
    route_df = predictions.merge(selected_routes, on=keys, how="inner", validate="many_to_one")
    if route_df.empty:
        st.warning("No backtest data available for this airport pair.")
        st.stop()

    st.subheader(f"{origin} → {destination}")
    view = st.radio("Forecast View", ["Individual Itinerary", "Airport-Pair Aggregate"], horizontal=True)

    # An itinerary is the full airport sequence in route_id, not just its endpoints.
    route_df["stop_count"] = route_df["route_id"].astype(str).str.count("-").sub(1).clip(lower=0)
    if view == "Individual Itinerary":
        paths = sorted(route_df["route_id"].dropna().unique().tolist(),
                       key=lambda x: (max(x.count("-") - 1, 0), x))
        path = st.selectbox(
            "Itinerary (full flight path)", paths,
            format_func=lambda x: f"{x}  ·  " + ("Nonstop" if x.count("-") == 1 else f"{x.count('-') - 1} stop(s)"),
        )
        route_df = route_df.loc[route_df["route_id"] == path].copy()
        st.caption(f"Selected itinerary: **{path}**. The route ID represents a path through airports; it is not necessarily a nonstop flight.")
        # Preserve model grain; do not collapse different market identifiers silently.
        if route_df.duplicated(["Year", "Quarter"]).any():
            st.error("The selected itinerary still has multiple market-level records for one quarter. Select a market identifier before comparing predictions.")
            st.dataframe(route_df.loc[route_df.duplicated(["Year", "Quarter"], keep=False),
                                      ["Year", "Quarter", *keys]], hide_index=True, use_container_width=True)
            st.stop()
        route_df["itinerary_count"] = 1
    else:
        # Aggregate predictions first, then recompute errors from quarterly totals.
        # Counts refer to represented paths, not necessarily all passengers in the O&D market.
        st.caption("Airport-pair aggregate: sums all **itineraries represented in the test predictions** for each quarter. This is not a verified full-market passenger total.")
        route_df = (route_df.groupby(["Year", "Quarter"], as_index=False)
                    .agg(target_passengers=("target_passengers", "sum"),
                         rf_predicted=("rf_predicted", "sum"),
                         naive_predicted=("naive_predicted", "sum"),
                         itinerary_count=("route_id", "nunique")))

    route_df = add_period(route_df)
    all_periods = route_df["Period"].tolist()
    if len(all_periods) > 1:
        start, end = st.select_slider("Test-quarter range", options=all_periods,
                                     value=(all_periods[0], all_periods[-1]),
                                     key="forecast_quarter_range")
        route_df = route_df.iloc[all_periods.index(start):all_periods.index(end) + 1].copy()
    if route_df.empty:
        st.warning("No data in selected period.")
        st.stop()

    actual = route_df["target_passengers"].to_numpy(dtype=float)
    rf_pred = route_df["rf_predicted"].to_numpy(dtype=float)
    naive_pred = route_df["naive_predicted"].to_numpy(dtype=float)
    route_df["RF Absolute Error"] = np.abs(actual - rf_pred)
    route_df["Naive Absolute Error"] = np.abs(actual - naive_pred)
    rf_mae = float(route_df["RF Absolute Error"].mean())
    naive_mae = float(route_df["Naive Absolute Error"].mean())
    rf_rmse = float(np.sqrt(np.mean((actual - rf_pred) ** 2)))
    naive_rmse = float(np.sqrt(np.mean((actual - naive_pred) ** 2)))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Random Forest MAE", f"{rf_mae:,.2f}")
    c2.metric("Seasonal Naive MAE", f"{naive_mae:,.2f}")
    c3.metric("Random Forest RMSE", f"{rf_rmse:,.2f}")
    c4.metric("Seasonal Naive RMSE", f"{naive_rmse:,.2f}")
    st.caption("Metrics are recomputed from the visible quarterly actuals and predictions. Lower error is better.")

    st.subheader("Actual vs Predicted Passenger Demand")
    fig = go.Figure()
    for name, column, color, dash in [
        ("Actual", "target_passengers", DARK, "solid"),
        ("Random Forest", "rf_predicted", BLUE, "solid"),
        ("Seasonal Naive", "naive_predicted", ORANGE, "dash"),
    ]:
        fig.add_trace(go.Scatter(x=route_df["Date"], y=route_df[column],
                                 mode="lines+markers", name=name,
                                 line=dict(color=color, width=3, dash=dash)))
    st.plotly_chart(polish(fig, 460, y_title="DB1B passenger volume"), use_container_width=True)

    st.subheader("Absolute Forecast Error by Quarter")
    error_fig = go.Figure()
    error_fig.add_trace(go.Bar(x=route_df["Period"], y=route_df["RF Absolute Error"],
                               name="Random Forest", marker_color=BLUE))
    error_fig.add_trace(go.Bar(x=route_df["Period"], y=route_df["Naive Absolute Error"],
                               name="Seasonal Naive", marker_color=ORANGE))
    error_fig.update_layout(barmode="group")
    st.plotly_chart(polish(error_fig, 350, y_title="Absolute error"), use_container_width=True)

    if view == "Airport-Pair Aggregate":
        st.subheader("Included Itineraries per Quarter")
        count_fig = px.bar(route_df, x="Period", y="itinerary_count", text="itinerary_count")
        st.plotly_chart(polish(count_fig, 300, y_title="Distinct itineraries"), use_container_width=True)
        st.caption("Quarterly coverage can change; do not interpret absent itineraries as observed zero demand.")

    st.subheader("Model Comparison")
    if np.isclose(rf_mae, naive_mae):
        st.info("Both models have approximately equal MAE for the selected view and quarters.")
    elif rf_mae < naive_mae:
        pct = (naive_mae - rf_mae) / naive_mae * 100 if naive_mae else 0
        st.success(f"Random Forest has lower MAE than Seasonal Naive ({pct:.1f}% lower).")
    else:
        pct = (rf_mae - naive_mae) / rf_mae * 100 if rf_mae else 0
        st.info(f"Seasonal Naive has lower MAE than Random Forest ({pct:.1f}% lower).")

    st.subheader("Quarterly Forecast Results")
    columns = ["Year", "Quarter", "target_passengers", "rf_predicted", "naive_predicted",
               "RF Absolute Error", "Naive Absolute Error", "itinerary_count"]
    display_df = route_df[columns].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    suffix = "aggregate" if view == "Airport-Pair Aggregate" else path.replace("-", "_")
    st.download_button("⬇️ Download Forecast Results (CSV)", display_df.to_csv(index=False),
                       f"{origin}_{destination}_{suffix}_backtest.csv", "text/csv")

elif page == "Model Insights":
    st.title("Model Insights")
    st.write("Evaluate test-set performance, forecast errors, robustness, and feature contributions.")
    tab1, tab2, tab3, tab4 = st.tabs(["Model Comparison", "Error Analysis", "Model Robustness", "Explainability"])
    with tab1:
        st.header("Forecasting Model Comparison")
        st.caption("All metrics are from the 2024 Q1–2025 Q2 test set; lower MAE/RMSE and higher R² are better.")
        metrics = load_results_csv("random_forest_metrics.csv")
        if metrics is None:
            st.error("Model metrics file not found.")
            st.stop()
        model_results = metrics.loc[
            metrics["Split"].astype(str).str.startswith("Test"),
            ["Model", "MAE", "RMSE", "R2"]
        ].copy()
        if model_results.empty:
            st.error("No test-set metrics available.")
            st.stop()
        st.dataframe(model_results.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"}), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(model_results.sort_values("MAE"), x="Model", y="MAE", text="MAE", color_discrete_sequence=[BLUE])
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(polish(fig, 420, x_title=None, y_title="MAE"), use_container_width=True)
        with c2:
            fig = px.bar(model_results.sort_values("RMSE"), x="Model", y="RMSE", text="RMSE", color_discrete_sequence=[ORANGE])
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(polish(fig, 420, x_title=None, y_title="RMSE"), use_container_width=True)
        best_mae = model_results.loc[model_results["MAE"].idxmin(), "Model"]
        best_rmse = model_results.loc[model_results["RMSE"].idxmin(), "Model"]
        st.info(f"Lowest MAE: {best_mae}. Lowest RMSE: {best_rmse}.")
    with tab2:
        st.header("Forecast Error Analysis")
        overall = load_results_csv("overall_error_metrics.csv")
        yearly = load_results_csv("error_by_year.csv")
        route_size = load_results_csv("error_by_route_size.csv")
        st.subheader("Overall Error Metrics")
        if overall is None:
            st.warning("overall_error_metrics.csv not found.")
        else:
            overall = overall.copy()
            if len(overall.columns) == 2:
                overall.columns = ["Metric", "Value"]
            st.dataframe(overall, use_container_width=True, hide_index=True)
        st.subheader("Forecast Error by Route Size")
        if route_size is not None:
            st.dataframe(route_size, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            display_saved_figure("mae_by_route_size.png", "MAE across route-size categories")
        with c2:
            display_saved_figure("rmse_by_route_size.png", "RMSE across route-size categories")
        st.caption("Compare route-size error metrics above to evaluate performance across demand segments.")
        st.divider()
        st.subheader("Forecast Error by Year")
        if yearly is not None:
            st.dataframe(yearly, use_container_width=True, hide_index=True)
        display_saved_figure("mae_by_year.png", "Forecast MAE by test year")
        st.divider()
        st.subheader("Absolute Error Distribution")
        display_saved_figure("error_distribution.png", "Absolute prediction error distribution")
    with tab3:
        st.header("Walk-Forward Validation")
        st.write("Expanding-window validation tests robustness through the COVID-19 disruption and subsequent recovery.")
        walk_forward = load_results_csv("walk_forward_results.csv")
        if walk_forward is None:
            st.warning("walk_forward_results.csv not found.")
        else:
            st.dataframe(walk_forward, use_container_width=True, hide_index=True)
            year_col = "Validation_Year" if "Validation_Year" in walk_forward.columns else "ValidationYear" if "ValidationYear" in walk_forward.columns else None
            if year_col is None:
                st.warning("Validation year column not found in walk-forward CSV.")
            else:
                for metric in ("MAE", "RMSE"):
                    st.subheader(f"{metric} Across Validation Years")
                    columns = [f"RF_{metric}", f"Naive_{metric}"]
                    if not set(columns).issubset(walk_forward.columns):
                        st.warning(f"Missing {metric} columns in walk-forward CSV.")
                        continue
                    plot_df = walk_forward.melt(id_vars=[year_col], value_vars=columns, var_name="Model", value_name=metric)
                    plot_df.Model = plot_df.Model.replace({columns[0]: "Random Forest", columns[1]: "Seasonal Naive"})
                    fig = px.line(plot_df, x=year_col, y=metric, color="Model", markers=True,
                                  color_discrete_map={"Random Forest": BLUE, "Seasonal Naive": ORANGE})
                    st.plotly_chart(polish(fig, 430, x_title="Validation year", y_title=metric), use_container_width=True)
            st.caption("Compare validation-year errors to assess performance across different market conditions.")
    with tab4:
        st.header("Model Explainability")
        st.write("SHAP attributes Random Forest predictions to input features; these explanations are not causal estimates.")
        st.subheader("Global Feature Importance")
        display_saved_figure("shap_feature_importance.png", "Mean absolute SHAP values")
        st.markdown("Historical passenger demand is the dominant source of predictive information.")
        st.subheader("SHAP Summary")
        display_saved_figure("shap_beeswarm.png", "SHAP feature contributions")
        st.subheader("Individual Forecast Explanation")
        display_saved_figure("shap_waterfall.png", "Example Random Forest prediction explanation")
        st.divider()
        st.header("Ablation Study")
        st.write("Does adding lagged airfare and city-market competition features improve forecasting?")
        ablation = load_results_csv("ablation_results.csv")
        if ablation is None:
            st.warning("ablation_results.csv not found.")
        else:
            st.dataframe(ablation, use_container_width=True, hide_index=True)
            if {"Model", "MAE", "RMSE"}.issubset(ablation.columns):
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(ablation, x="Model", y="MAE", text="MAE", color_discrete_sequence=[BLUE])
                    fig.update_xaxes(tickangle=-15)
                    st.plotly_chart(polish(fig, 400, x_title=None, y_title="MAE"), use_container_width=True)
                with c2:
                    fig = px.bar(ablation, x="Model", y="RMSE", text="RMSE", color_discrete_sequence=[ORANGE])
                    fig.update_xaxes(tickangle=-15)
                    st.plotly_chart(polish(fig, 400, x_title=None, y_title="RMSE"), use_container_width=True)
            if {"Model", "MAE", "RMSE"}.issubset(ablation.columns) and len(ablation) >= 2:
                baseline = ablation.loc[ablation["Model"] == "Demand History Only"]
                extended = ablation.loc[ablation["Model"] == "Demand + Market Conditions"]
                if not baseline.empty and not extended.empty:
                    b = baseline.iloc[0]
                    e = extended.iloc[0]
                    mae_gain = (b["MAE"] - e["MAE"]) / b["MAE"] * 100
                    rmse_gain = (b["RMSE"] - e["RMSE"]) / b["RMSE"] * 100
                    st.info(
                        f"Extended features (lagged fare, competition, and distance) "
                        f"change validation MAE by {-mae_gain:+.2f}% and "
                        f"RMSE by {-rmse_gain:+.2f}% relative to demand history only."
                    )
