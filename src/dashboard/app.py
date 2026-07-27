"""EV-Pulse Dashboard - Interactive Smart Charging Simulator.

This module provides a Streamlit-based web interface for running
EV charging load simulations and visualizing predictions.
"""

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# --- CONFIGURATION ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- PAGE SETUP ---
st.set_page_config(
    page_title="EV-Pulse Dashboard",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded",
)

st.title("⚡ EV-Pulse: Charging Load Simulator")
st.markdown("*A small day-ahead forecasting project for EV charging sites*")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Simulation Settings")

# Date Selection
today = datetime.now()
selected_date = st.sidebar.date_input(
    "📅 Target Date",
    value=today,
    help="Select the date for load prediction simulation",
)

# Weather Settings
st.sidebar.subheader("🌤️ Weather Conditions")
use_seasonal = st.sidebar.checkbox(
    "Use seasonal averages",
    value=True,
    help="Automatically use historical weather averages for the selected month",
)

if not use_seasonal:
    override_temp = st.sidebar.slider(
        "Temperature (°C)",
        min_value=-10,
        max_value=45,
        value=20,
        help="Set custom temperature for simulation",
    )
    override_sun = st.sidebar.slider(
        "Solar Radiation (W/m²)",
        min_value=0,
        max_value=1200,
        value=600,
        help="Set custom solar radiation level",
    )
else:
    override_temp = None
    override_sun = None
    st.sidebar.info("📅 Weather will be inferred from the selected month's historical data.")

# Infrastructure Settings
st.sidebar.subheader("🔌 Infrastructure Sizing")
ev_growth = st.sidebar.slider(
    "EV Fleet Growth (%)",
    min_value=0,
    max_value=200,
    value=0,
    help="Simulate increased EV adoption (scales predictions)",
)
grid_limit = st.sidebar.number_input(
    "Grid Capacity Limit (kW)",
    min_value=50,
    max_value=1000,
    value=150,
    help="Transformer/grid capacity threshold for warnings",
)

st.sidebar.divider()

# --- RUN SIMULATION ---
if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
    with st.spinner("Running the prediction model..."):
        try:
            # API Request
            payload = {
                "date": selected_date.strftime("%Y-%m-%d"),
                "override_temp": override_temp,
                "override_sun": override_sun,
            }

            response = requests.post(f"{API_URL}/simulate", json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Process response data
            df_res = pd.DataFrame(data["points"])
            df_res["datetime"] = pd.to_datetime(df_res["datetime"])
            df_res["hour"] = df_res["datetime"].dt.hour

            # Apply EV growth factor
            growth_factor = 1 + (ev_growth / 100)
            df_res["predicted_power_kw"] = df_res["predicted_power_kw"] * growth_factor

            # Recompute warnings locally: growth factor and custom grid limit
            # can differ from the API defaults
            df_res["is_peak_warning"] = df_res["predicted_power_kw"] > grid_limit

            # Calculate statistics with growth factor
            total_energy = sum(df_res["predicted_power_kw"]) / 4
            peak_power = df_res["predicted_power_kw"].max()
            avg_power = df_res["predicted_power_kw"].mean()
            warning_intervals = (df_res["predicted_power_kw"] > grid_limit).sum()

            summary = data.get("summary", {})
            weather = data.get("weather", {})

            # --- KEY PERFORMANCE INDICATORS ---
            st.subheader("📊 Simulation Results")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="⚡ Total Energy",
                    value=f"{total_energy:,.0f} kWh",
                    help="Total energy consumption for the day",
                )

            with col2:
                delta_pct = ((peak_power - grid_limit) / grid_limit) * 100
                st.metric(
                    label="📈 Peak Power",
                    value=f"{peak_power:,.1f} kW",
                    delta=f"{delta_pct:+.1f}% vs limit",
                    delta_color="inverse" if peak_power > grid_limit else "normal",
                    help="Maximum power draw during the day",
                )

            with col3:
                st.metric(
                    label="📉 Average Power",
                    value=f"{avg_power:,.1f} kW",
                    help="Mean power consumption across all intervals",
                )

            with col4:
                weather_display = f"{weather.get('temperature_c', 'N/A')}°C"
                st.metric(
                    label="🌡️ Temperature",
                    value=weather_display,
                    help=f"Source: {weather.get('source', 'unknown')}",
                )

            # Warning alert if capacity exceeded
            if warning_intervals > 0:
                st.error(
                    f"⚠️ **Grid Overload Warning**: {warning_intervals} intervals "
                    f"({warning_intervals * 15} minutes) exceed the {grid_limit} kW capacity limit!"
                )
            else:
                st.success("✅ All predictions are within grid capacity limits.")

            st.divider()

            # --- MAIN VISUALIZATION ---
            st.subheader("📈 Load Profile")

            fig_main = go.Figure()

            # Overload zone shading
            if peak_power > grid_limit:
                fig_main.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=grid_limit,
                    x1=1,
                    y1=peak_power * 1.1,
                    fillcolor="rgba(255, 87, 34, 0.15)",
                    line_width=0,
                    layer="below",
                )

            # Capacity limit line
            fig_main.add_hline(
                y=grid_limit,
                line_dash="dash",
                line_color="#ff5722",
                annotation_text=f"Capacity Limit ({grid_limit} kW)",
                annotation_position="top right",
            )

            # Average power line
            fig_main.add_hline(
                y=avg_power,
                line_dash="dot",
                line_color="#2196F3",
                annotation_text=f"Average ({avg_power:.0f} kW)",
                annotation_position="bottom right",
            )

            # Main load curve
            fig_main.add_trace(
                go.Scatter(
                    x=df_res["datetime"],
                    y=df_res["predicted_power_kw"],
                    mode="lines",
                    name="Predicted Load",
                    fill="tozeroy",
                    line=dict(color="#4CAF50", width=2),
                    fillcolor="rgba(76, 175, 80, 0.3)",
                    hovertemplate="<b>%{x|%H:%M}</b><br>Power: %{y:.1f} kW<extra></extra>",
                )
            )

            fig_main.update_layout(
                title=dict(
                    text=f"Power Consumption Forecast — {selected_date.strftime('%A, %B %d, %Y')}",
                    x=0.5,
                    xanchor="center",
                ),
                xaxis_title="Time",
                yaxis_title="Power (kW)",
                height=450,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=60, r=40, t=80, b=60),
            )

            st.plotly_chart(fig_main, use_container_width=True)

            # --- SECONDARY VISUALIZATIONS ---
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("🕐 Hourly Distribution")

                # Aggregate by hour
                hourly_df = df_res.groupby("hour")["predicted_power_kw"].mean().reset_index()
                hourly_df.columns = ["Hour", "Average Power (kW)"]

                # Color based on capacity
                hourly_df["Status"] = hourly_df["Average Power (kW)"].apply(
                    lambda x: "Over Capacity" if x > grid_limit else "Normal"
                )

                fig_hourly = px.bar(
                    hourly_df,
                    x="Hour",
                    y="Average Power (kW)",
                    color="Status",
                    color_discrete_map={"Normal": "#4CAF50", "Over Capacity": "#ff5722"},
                    title="Average Power by Hour",
                )

                fig_hourly.add_hline(
                    y=grid_limit,
                    line_dash="dash",
                    line_color="#ff5722",
                    annotation_text="Limit",
                )

                fig_hourly.update_layout(
                    xaxis=dict(tickmode="linear", dtick=2),
                    showlegend=False,
                    height=350,
                )

                st.plotly_chart(fig_hourly, use_container_width=True)

            with col_right:
                st.subheader("📊 Load Distribution")

                fig_hist = px.histogram(
                    df_res,
                    x="predicted_power_kw",
                    nbins=30,
                    title="Power Level Distribution",
                    labels={"predicted_power_kw": "Power (kW)", "count": "Frequency"},
                    color_discrete_sequence=["#2196F3"],
                )

                fig_hist.add_vline(
                    x=grid_limit,
                    line_dash="dash",
                    line_color="#ff5722",
                    annotation_text="Limit",
                )

                fig_hist.add_vline(
                    x=avg_power,
                    line_dash="dot",
                    line_color="#4CAF50",
                    annotation_text="Average",
                )

                fig_hist.update_layout(height=350)

                st.plotly_chart(fig_hist, use_container_width=True)

            # --- DATA TABLE ---
            with st.expander("📋 View Raw Data"):
                st.dataframe(
                    df_res[["datetime", "predicted_power_kw", "is_peak_warning"]]
                    .rename(
                        columns={
                            "datetime": "Timestamp",
                            "predicted_power_kw": "Power (kW)",
                            "is_peak_warning": "Warning",
                        }
                    )
                    .style.format({"Power (kW)": "{:.2f}"})
                    .map(
                        lambda x: "background-color: #ffcccc" if x else "",
                        subset=["Warning"],
                    ),
                    use_container_width=True,
                    height=400,
                )

                # Download button
                csv = df_res.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"ev_pulse_simulation_{selected_date}.csv",
                    mime="text/csv",
                )

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Please ensure the backend is running.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

else:
    # Welcome state
    st.info("👈 Configure simulation parameters in the sidebar and click **Run Simulation** to begin.")

    # Show API health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health.get("model_loaded"):
            st.success(f"✅ API Connected | Model Ready | Version {health.get('version', 'N/A')}")
        else:
            st.warning("⚠️ API Connected but model not loaded. Check server logs.")
    except requests.exceptions.RequestException:
        st.warning("⚠️ API not reachable. Start the backend with: `uvicorn src.api.main:app --reload`")
