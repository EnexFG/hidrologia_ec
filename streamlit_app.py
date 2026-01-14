import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import ConciseDateFormatter

st.set_page_config(page_title="Hidrología EC", layout="wide")

meses_abrev = {
    '01': 'Ene', '02': 'Feb', '03': 'Mar', '04': 'Abr',
    '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Ago',
    '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dic'
}

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)

st.title("Explorador de caudales — Cotas por estación y año")

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

# Preparación base
df_plot = df[df["year"].isin(selected_years)][["year", "month", "day", station]].copy()
df_plot[station] = df_plot[station].replace(0, pd.NA)

# Normalizar month/day a string con cero a la izquierda
df_plot["month"] = df_plot["month"].astype(int).astype(str).str.zfill(2)
df_plot["day"] = df_plot["day"].astype(int).astype(str).str.zfill(2)

# MultiIndex para el eje X: (mes_abrev, dia)
df_plot["month_name"] = df_plot["month"].map(meses_abrev)
df_plot["day_int"] = df_plot["day"].astype(int)

# Orden natural por mes/día usando un "doy" auxiliar (año no bisiesto)
md = df_plot["month"] + "-" + df_plot["day"]
df_plot["date_dummy"] = pd.to_datetime(md, format="%m-%d", errors="coerce")
df_plot["doy"] = df_plot["date_dummy"].dt.dayofyear

# Pivot por doy (para graficar fácil y consistente)
pivot = df_plot.pivot_table(index="doy", columns="year", values=station).sort_index()

# También construimos el MultiIndex (mes,día) para el eje X “semántico”
# (mapeando doy -> (mes_abrev, dia))
doy_to_md = (
    df_plot.dropna(subset=["doy"])
           .drop_duplicates(subset=["doy"])
           .set_index("doy")[["month", "day"]]
           .sort_index()
)

# Aseguramos cobertura para los doy del pivot
doy_index = pivot.index.astype(int)
md_for_doy = doy_to_md.reindex(doy_index)

# Si faltan algunos, hacemos fallback con fecha dummy desde doy (año 2001)
missing = md_for_doy["month"].isna()
if missing.any():
    fallback_dates = pd.to_datetime(doy_index[missing].astype(str), format="%j", errors="coerce") + pd.DateOffset(years=2001-1900)
    # el fallback puede fallar dependiendo de pandas; por eso hacemos algo más robusto:
    # usamos un rango base y indexamos por doy
    base = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    base_map = pd.DataFrame({
        "doy": base.dayofyear,
        "month": base.month.astype(int).astype(str).str.zfill(2),
        "day": base.day.astype(int).astype(str).str.zfill(2)
    }).set_index("doy")
    md_for_doy.loc[missing, ["month", "day"]] = base_map.reindex(doy_index[missing])[["month", "day"]].values

x_month = md_for_doy["month"].astype(str).str.zfill(2).tolist()
x_day = md_for_doy["day"].astype(str).str.zfill(2).tolist()
x_labels_monthday = [f"{meses_abrev[m]}-{d}" for m, d in zip(x_month, x_day)]

# Ticks: solo inicio de cada mes para no abombar el eje
month_start_positions = []
month_start_labels = []
seen = set()
for i, (m, d) in enumerate(zip(x_month, x_day)):
    if m not in seen:  # primer día que aparece ese mes en el eje
        seen.add(m)
        month_start_positions.append(i)
        month_start_labels.append(f"{meses_abrev[m]}-{d}")

# ------------------------
# GRÁFICO 1
# ------------------------
st.subheader("Gráfico 1 — Comparación interanual (cada línea = un año)")

fig1, ax1 = plt.subplots(figsize=(11, 5))
xpos = np.arange(len(pivot.index))

for y in selected_years:
    if y in pivot.columns:
        ax1.plot(xpos, pivot[y].values, label=str(y))

ax1.set_xlabel("Mes–día")
ax1.set_ylabel(station)
ax1.set_xticks(month_start_positions)
ax1.set_xticklabels(month_start_labels)
ax1.legend()
ax1.grid(True)

# --- Eje X estilo Excel: días (ticks) + meses (segunda fila) ---

# xpos ya es np.arange(len(pivot.index))
# x_month y x_day ya los tienes como listas (strings '01'..'12' y '01'..'31')

# 1) Ticks de DÍAS (elige pocos para no abombar)
dias_mostrar = {1, 10, 22}  # puedes ajustar: {1, 11, 21} etc.
day_tick_pos = [i for i, d in enumerate(x_day) if int(d) in dias_mostrar]
day_tick_lab = [str(int(x_day[i])) for i in day_tick_pos]

ax1.set_xticks(day_tick_pos)
ax1.set_xticklabels(day_tick_lab)
ax1.tick_params(axis="x", pad=2)  # acerca los días al eje

# 2) Segundo eje X ABAJO para los MESES (tipo Excel)
secax = ax1.secondary_xaxis('bottom')
secax.spines['bottom'].set_position(('outward', 22))  # baja la “fila” de meses

# Posición de inicio de cada mes (primer punto donde aparece el mes)
month_start_pos = []
month_labels = []
seen = set()
for i, m in enumerate(x_month):
    if m not in seen:
        seen.add(m)
        month_start_pos.append(i)
        month_labels.append(meses_abrev[m])  # 'Ene', 'Feb', ...

secax.set_xticks(month_start_pos)
secax.set_xticklabels(month_labels)

# Limpieza visual
ax1.set_xlabel("")      # el eje principal ya muestra días
secax.set_xlabel("")    # sin título para que parezca Excel

st.pyplot(fig1)

st.markdown("---")

# ------------------------
# GRÁFICO 2 (una sola línea continua, segmentos por año con colores distintos)
# ------------------------
st.subheader("Gráfico 2 — Serie continua (una sola línea, colores por año)")

df_ts = df_plot.copy()

# Construir fecha real
df_ts["date"] = pd.to_datetime(
    df_ts["year"].astype(int).astype(str) + "-" +
    df_ts["month"].astype(str).str.zfill(2) + "-" +
    df_ts["day"].astype(str).str.zfill(2),
    errors="coerce"
)

df_ts = df_ts.dropna(subset=["date"]).sort_values("date")

# Serie completa (una sola)
x_dates = df_ts["date"].to_numpy()
y_vals = df_ts[station].to_numpy(dtype=float)
year_vals = df_ts["year"].to_numpy()

# Convertir fechas a formato numérico (matplotlib)
x_num = mdates.date2num(x_dates)

# Segmentar en tramos consecutivos (evitar NaN y saltos grandes de fecha)
valid = np.isfinite(y_vals)

segments = []
segment_years = []

for i in range(len(x_num) - 1):
    if not (valid[i] and valid[i+1]):
        continue

    # Si hay saltos grandes (por huecos), corta la línea (opcional pero recomendado)
    # Por ejemplo: si falta data y se brinca > 7 días, no unimos esos puntos.
    if (x_num[i+1] - x_num[i]) > 7:
        continue

    segments.append([[x_num[i], y_vals[i]], [x_num[i+1], y_vals[i+1]]])
    segment_years.append(year_vals[i])

segments = np.asarray(segments, dtype=float)

unique_years = sorted(set(segment_years))
year_to_idx = {yy: i for i, yy in enumerate(unique_years)}
cvals = np.array([year_to_idx[yy] for yy in segment_years], dtype=float)

cmap = plt.get_cmap("tab10", max(len(unique_years), 1))
lc = LineCollection(segments, cmap=cmap)
lc.set_array(cvals)
lc.set_linewidth(2.0)

fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.add_collection(lc)

ax2.set_xlim(x_num.min(), x_num.max())

finite_y = y_vals[np.isfinite(y_vals)]
if finite_y.size > 0:
    pad = 0.05 * (finite_y.max() - finite_y.min() + 1e-9)
    ax2.set_ylim(finite_y.min() - pad, finite_y.max() + pad)

ax2.set_xlabel("Día (fecha)")
ax2.set_ylabel(station)
ax2.grid(True)

# Formato bonito de fechas (auto)
locator = mdates.AutoDateLocator()
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(ConciseDateFormatter(locator))

# Leyenda por año
handles = [Line2D([0], [0], color=cmap(i), lw=2, label=str(yy)) for i, yy in enumerate(unique_years)]
ax2.legend(handles=handles, title="Año")

st.pyplot(fig2)