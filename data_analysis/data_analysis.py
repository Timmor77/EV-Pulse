import pandas as pd
import plotly.express as px
import streamlit as st

# streamlit run .\data_analysis\data_analysis.py

# Config page
st.set_page_config(page_title="EV-Pulse Data Explorer", layout="wide")

st.title("⚡ EV-Pulse: Data Exploration")


# Optimized data loading with cache
@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/acn_timeseries_15min.parquet")
    return df


df = load_data()

# Sidebar
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    value=[df["datetime"].min(), df["datetime"].max()],
    min_value=df["datetime"].min(),
    max_value=df["datetime"].max(),
)

# Filtrage
mask = (df["datetime"].dt.date >= date_range[0]) & (df["datetime"].dt.date <= date_range[1])
df_filtered = df.loc[mask]

# --- KPI ---
col1, col2, col3 = st.columns(3)
col1.metric("Peak Power (Max Load)", f"{df_filtered['power_kw'].max():.2f} kW")
col2.metric("Average Load", f"{df_filtered['power_kw'].mean():.2f} kW")
col3.metric("Max Active Chargers", f"{df_filtered['active_chargers'].max()} chargers")

# --- CHART 1: TIME SERIES ---
st.subheader("📈 Load Curve (Total Power)")
fig_ts = px.line(
    df_filtered,
    x="datetime",
    y="power_kw",
    title="Power consumption over time",
)
st.plotly_chart(fig_ts, use_container_width=True)

# --- CHART 2: DAILY AVERAGE PROFILE ---
st.subheader("🕓 Average Profile: When do people charge?")
# Extract hour
df_filtered["hour"] = df_filtered["datetime"].dt.hour
hourly_profile = df_filtered.groupby("hour")["power_kw"].mean().reset_index()

fig_profile = px.bar(
    hourly_profile,
    x="hour",
    y="power_kw",
    title="Average Power by Hour of Day",
    labels={"hour": "Hour (0-23)", "power_kw": "Average Power (kW)"},
)
st.plotly_chart(fig_profile, use_container_width=True)

# --- QUALITY ANALYSIS ---
st.markdown("---")
st.subheader("🕵️‍♂️ Data Consistency Check")

# Zoom on a typical week (first week of selection)
st.write("Zoom on the first 7 days of selection (to observe Day/Night cycles):")
st.plotly_chart(
    px.line(df_filtered.head(24 * 4 * 7), x="datetime", y="power_kw"),
    use_container_width=True,
)
