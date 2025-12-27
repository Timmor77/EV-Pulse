import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="EV-Pulse Dashboard", layout="wide", page_icon="⚡")

st.title("⚡ EV-Pulse : Smart Charging Simulator")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Settings")

# 1. Date (Default = Today)
today = datetime.now()
selected_date = st.sidebar.date_input("Target Date", value=today)

# 2. Smart Weather
st.sidebar.subheader("Weather Conditions")
use_seasonal = st.sidebar.checkbox("Use seasonal averages", value=True)

if not use_seasonal:
    override_temp = st.sidebar.slider("Temperature (°C)", -5, 45, 20)
    override_sun = st.sidebar.slider("Solar Radiation (W/m²)", 0, 1000, 600)
else:
    override_temp = None
    override_sun = None
    st.sidebar.info("📅 Weather will be automatically inferred from the selected month.")

st.sidebar.subheader("Sizing")
ev_growth = st.sidebar.slider("EV Fleet Growth (%)", 0, 200, 0)
grid_limit = st.sidebar.number_input("Transformer Limit (kW)", value=150)

# --- SIMULATION ---
if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Calculating..."):
        try:
            payload = {
                "date": selected_date.strftime("%Y-%m-%d"),
                "override_temp": override_temp,
                "override_sun": override_sun,
            }

            response = requests.post(f"{API_URL}/simulate", json=payload)
            response.raise_for_status()
            data = response.json()

            # ... (Rest of display code remains the same) ...
            # COPY PASTE THE REST OF THE PREVIOUS FILE HERE (df_res processing, KPI, Chart)

            # Processing block:
            df_res = pd.DataFrame(data["points"])
            df_res["datetime"] = pd.to_datetime(df_res["datetime"])
            growth_factor = 1 + (ev_growth / 100)
            df_res["predicted_power_kw"] = df_res["predicted_power_kw"] * growth_factor

            total_energy = sum(df_res["predicted_power_kw"]) / 4
            peak_power = df_res["predicted_power_kw"].max()

            # KPI
            col1, col2, col3 = st.columns(3)
            col1.metric("⚡ Total Energy", f"{total_energy:.0f} kWh")

            d_col = "inverse" if peak_power > grid_limit else "normal"
            col2.metric(
                "📈 Peak Power",
                f"{peak_power:.1f} kW",
                delta=f"{peak_power - grid_limit:.1f}",
                delta_color=d_col,
            )

            # Display weather used (from API message or local logic)
            weather_msg = "Seasonal" if use_seasonal else f"{override_temp}°C"
            col3.metric("☀️ Weather", weather_msg)

            # Graphique
            fig = go.Figure()
            if peak_power > grid_limit:
                fig.add_shape(
                    type="rect",
                    xref="paper",
                    yref="y",
                    x0=0,
                    y0=grid_limit,
                    x1=1,
                    y1=max(peak_power, grid_limit) * 1.1,
                    fillcolor="red",
                    opacity=0.1,
                    line_width=0,
                )

            fig.add_hline(
                y=grid_limit,
                line_dash="dash",
                line_color="red",
                annotation_text="Limit",
            )
            fig.add_trace(
                go.Scatter(
                    x=df_res["datetime"],
                    y=df_res["predicted_power_kw"],
                    mode="lines",
                    name="Load",
                    fill="tozeroy",
                    line=dict(color="#4CAF50", width=3),
                )
            )

            fig.update_layout(
                title=f"Load Profile for {selected_date.strftime('%Y-%m-%d')}",
                yaxis_title="kW",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"API Error: {e}")
