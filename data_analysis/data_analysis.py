import pandas as pd
import plotly.express as px
import streamlit as st

# streamlit run .\data_analysis\data_analysis.py

# Config page
st.set_page_config(page_title="EV-Pulse Data Explorer", layout="wide")

st.title("⚡ EV-Pulse : Exploration des Données")


# Chargement optimisé avec cache
@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/acn_timeseries_15min.parquet")
    return df


df = load_data()

# Sidebar
st.sidebar.header("Filtres")
date_range = st.sidebar.date_input(
    "Période",
    value=[df["datetime"].min(), df["datetime"].max()],
    min_value=df["datetime"].min(),
    max_value=df["datetime"].max(),
)

# Filtrage
mask = (df["datetime"].dt.date >= date_range[0]) & (
    df["datetime"].dt.date <= date_range[1]
)
df_filtered = df.loc[mask]

# --- KPI ---
col1, col2, col3 = st.columns(3)
col1.metric("Pic de Puissance (Max Load)", f"{df_filtered['power_kw'].max():.2f} kW")
col2.metric("Moyenne Charge", f"{df_filtered['power_kw'].mean():.2f} kW")
col3.metric("Bornes Actives Max", f"{df_filtered['active_chargers'].max()} bornes")

# --- GRAPHIQUE 1 : SÉRIE TEMPORELLE ---
st.subheader("📈 Courbe de Charge (Puissance Totale)")
fig_ts = px.line(
    df_filtered,
    x="datetime",
    y="power_kw",
    title="Puissance consommée au cours du temps",
)
st.plotly_chart(fig_ts, use_container_width=True)

# --- GRAPHIQUE 2 : PROFIL MOYEN JOURNALIER ---
st.subheader("🕓 Profil Moyen : À quelle heure charge-t-on ?")
# Extraction de l'heure
df_filtered["hour"] = df_filtered["datetime"].dt.hour
hourly_profile = df_filtered.groupby("hour")["power_kw"].mean().reset_index()

fig_profile = px.bar(
    hourly_profile,
    x="hour",
    y="power_kw",
    title="Puissance Moyenne par Heure de la Journée",
    labels={"hour": "Heure (0-23)", "power_kw": "Puissance Moyenne (kW)"},
)
st.plotly_chart(fig_profile, use_container_width=True)

# --- ANALYSE DE QUALITÉ ---
st.markdown("---")
st.subheader("🕵️‍♂️ Vérification de la cohérence")

# Zoom sur une semaine type (la première de la sélection)
st.write(
    "Zoom sur les 7 premiers jours de la sélection (pour voir les cycles Jour/Nuit) :"
)
st.plotly_chart(
    px.line(df_filtered.head(24 * 4 * 7), x="datetime", y="power_kw"),
    use_container_width=True,
)
