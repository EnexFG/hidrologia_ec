import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hidrología EC", layout="wide")

meses_map = {
    '01': 'Enero', '02': 'Febrero', '03': 'Marzo', '04': 'Abril',
    '05': 'Mayo', '06': 'Junio', '07': 'Julio', '08': 'Agosto',
    '09': 'Septiembre', '10': 'Octubre', '11': 'Noviembre', '12': 'Diciembre'
}

# (abreviaciones para el eje X sin abombar)
meses_abrev = {
    '01': 'Ene', '02': 'Feb', '03': 'Mar', '04': 'Abr',
    '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Ago',
    '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dic'
}

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)

st.title("Explorador de caudales — Cotas por estación y año")

# Carga fija: NO mostramos el nombre del archivo en la UI
DATA_PATH = "data_caudales_diario.pickle"

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"No se pudo cargar el pickle '{DATA_PATH}': {e}")
    st.stop()

# Columnas esperadas
cols = list(df.columns)
value_cols = [c for c in cols if c not in ("year", "month", "day")]
if not value_cols:
    st.error("No se encontraron columnas de valores (esperadas aparte de year, month, day).")
    st.write("Columnas detectadas:", cols)
    st.stop()

station = st.selectbox("Estación (columna de valores)", value_cols)

years = sorted(df["year"].dropna().unique())
selected_years = st.multiselect(
    "Años a mostrar",
    years,
    default=years[-5:] if len(years) > 5 else years
)

if not selected_years:
    st.info("Selecciona al menos un año.")
    st.stop()

# Filtrar y preparar para graficar
df_plot = df[df["year"].isin(selected_years)][["year", "month", "day", station]].copy()

# Reemplazar ceros por NaN
df_plot[station] = df_plot[station].replace(0, pd.NA)

# Crear doy (día del año)
md = df_plot["month"].astype(str).str.zfill(2) + "-" + df_plot["day"].astype(str).str.zfill(2)
df_plot["date_dummy"] = pd.to_datetime(md, format="%m-%d", errors="coerce")
df_plot["doy"] = df_plot["date_dummy"].dt.dayofyear

pivot = df_plot.pivot_table(index="doy", columns="year", values=station).sort_index()

# ============
# GRÁFICO 1: igual al que te gusta, pero eje X con MESES (12 ticks)
# ============
# Usamos un año no-bisiesto para calcular el doy del inicio de cada mes
month_starts = pd.date_range("2001-01-01", "2001-12-01", freq="MS")
month_start_doys = month_starts.dayofyear.tolist()
month_labels = [meses_abrev[f"{m:02d}"] for m in range(1, 13)]

fig1, ax1 = plt.subplots(figsize=(10, 5))
for y in selected_years:
    if y in pivot.columns:
        ax1.plot(pivot.index, pivot[y], label=str(y))

ax1.set_xlabel("Mes")
ax1.set_ylabel(station)
ax1.set_xticks(month_start_doys)
ax1.set_xticklabels(month_labels)
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

st.markdown("---")

# ============
# GRÁFICO 2: el gráfico de la versión anterior (doy “normal”, sin meses)
# ============
fig2, ax2 = plt.subplots(figsize=(10, 5))
for y in selected_years:
    if y in pivot.columns:
        ax2.plot(pivot.index, pivot[y], label=str(y))

ax2.set_xlabel("Día del año (ordenado por mes/día)")
ax2.set_ylabel(station)
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)