import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

st.set_page_config(page_title="Hidrología EC", layout="wide")

meses_abrev = {
    "01": "Ene", "02": "Feb", "03": "Mar", "04": "Abr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Ago",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dic"
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

# =========================
# Preparación base
# =========================
df_plot = df[df["year"].isin(selected_years)][["year", "month", "day", station]].copy()
df_plot[station] = df_plot[station].replace(0, pd.NA)

# Normalizar month/day
df_plot["year"] = df_plot["year"].astype(int)
df_plot["month"] = df_plot["month"].astype(int)
df_plot["day"] = df_plot["day"].astype(int)

# Fecha real para Gráfico 2
df_plot["date"] = pd.to_datetime(
    dict(year=df_plot["year"], month=df_plot["month"], day=df_plot["day"]),
    errors="coerce"
)

# doy auxiliar para Gráfico 1 (comparación interanual por mes/día)
df_plot["doy"] = df_plot["date"].dt.dayofyear

# Pivot por doy: cada columna un año
pivot = df_plot.pivot_table(index="doy", columns="year", values=station).sort_index()

# Mapeo doy -> (mes, día) usando un año base (bisiesto si hace falta)
max_doy = int(pivot.index.max()) if len(pivot.index) else 365
base_year = 2000 if max_doy == 366 else 2001
base = pd.date_range(f"{base_year}-01-01", f"{base_year}-12-31", freq="D")
base_map = pd.DataFrame({
    "doy": base.dayofyear,
    "month": base.month.astype(int).astype(str).str.zfill(2),
    "day": base.day.astype(int)
}).set_index("doy")

md_for_doy = base_map.reindex(pivot.index.astype(int))
x_month = md_for_doy["month"].tolist()
x_day = md_for_doy["day"].tolist()

# =========================
# GRÁFICO 1
# Cada línea = un año
# Eje X tipo Excel: días (ticks) + meses en una segunda fila abajo
# =========================
st.subheader("Gráfico 1 — Cada línea es un año (eje X: mes + días)")

xpos = np.arange(len(pivot.index))

fig1, ax1 = plt.subplots(figsize=(11, 5))
for y in sorted(selected_years):
    if y in pivot.columns:
        ax1.plot(xpos, pivot[y].values, label=str(y))

ax1.set_ylabel(station)
ax1.grid(True)
ax1.legend()

# Ticks de DÍAS (para no saturar)
dias_mostrar = {1, 8, 15, 22}
day_tick_pos = [i for i, d in enumerate(x_day) if int(d) in dias_mostrar]
day_tick_lab = [str(int(x_day[i])) for i in day_tick_pos]

ax1.set_xticks(day_tick_pos)
ax1.set_xticklabels(day_tick_lab)
ax1.tick_params(axis="x", pad=2)
ax1.set_xlabel("")  # sin título (parecido a Excel)

# Segunda fila: MESES
secax = ax1.secondary_xaxis("bottom")
secax.spines["bottom"].set_position(("outward", 22))

month_start_pos = []
month_labels = []
seen = set()
for i, m in enumerate(x_month):
    if m not in seen:
        seen.add(m)
        month_start_pos.append(i)
        month_labels.append(meses_abrev.get(m, m))

secax.set_xticks(month_start_pos)
secax.set_xticklabels(month_labels)
secax.set_xlabel("")

st.pyplot(fig1)

st.markdown("---")

# =========================
# GRÁFICO 2
# Una sola línea continua con colores por año
# Eje X = FECHA REAL
# =========================
st.subheader("Gráfico 2 — Una sola línea continua (eje X: fecha; colores por año)")

# Construir serie diaria por fecha y año (evita duplicados)
df_ts = (
    df_plot.dropna(subset=["date"])
           .groupby(["date", "year"], as_index=False)[station]
           .mean()
           .sort_values("date")
)

# Para garantizar continuidad "real", armamos un vector completo en orden cronológico
# (simple: concatenar años seleccionados ordenados; si quieres estrictamente por fecha global, ya está por date)
years_sorted = sorted(selected_years)
parts = []
for y in years_sorted:
    part = df_ts[df_ts["year"] == y][["date", station, "year"]].sort_values("date")
    parts.append(part)

df_concat = pd.concat(parts, ignore_index=True).sort_values("date")
df_concat = df_concat.reset_index(drop=True)

if df_concat.empty:
    st.info("No hay datos para graficar en los años seleccionados.")
    st.stop()

x_dates = df_concat["date"].to_numpy()
y_vals = df_concat[station].to_numpy(dtype=float)
y_year = df_concat["year"].to_numpy(dtype=int)

# Convertir fechas a números de Matplotlib
x_num = mdates.date2num(x_dates)

# Crear segmentos y colorear por año (multicolor en una sola línea)
valid = np.isfinite(y_vals)
points = np.column_stack([x_num, y_vals])

segments = []
segment_years = []

for i in range(len(points) - 1):
    if not (valid[i] and valid[i + 1]):
        continue

    # Romper si hay saltos grandes de fechas (para no unir huecos raros)
    if (x_dates[i + 1] - x_dates[i]) > np.timedelta64(2, "D"):
        continue

    segments.append([points[i], points[i + 1]])
    segment_years.append(y_year[i])

segments = np.asarray(segments, dtype=float)

unique_years = years_sorted
year_to_idx = {yy: i for i, yy in enumerate(unique_years)}
cvals = np.array([year_to_idx[yy] for yy in segment_years], dtype=float)

cmap = plt.get_cmap("tab10", max(len(unique_years), 1))

fig2, ax2 = plt.subplots(figsize=(11, 5))

if len(segments) > 0:
    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(cvals)
    lc.set_linewidth(2.0)
    ax2.add_collection(lc)

    ax2.set_xlim(x_num.min(), x_num.max())

    finite_y = y_vals[np.isfinite(y_vals)]
    if finite_y.size > 0:
        pad = 0.05 * (finite_y.max() - finite_y.min() + 1e-9)
        ax2.set_ylim(finite_y.min() - pad, finite_y.max() + pad)
else:
    # Fallback si no se pudo crear segmentos (por muchos NaN, etc.)
    ax2.plot(x_dates, y_vals, linewidth=2.0)

ax2.set_ylabel(station)
ax2.grid(True)

# Formato del eje X como fecha (automático y limpio)
locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
formatter = mdates.ConciseDateFormatter(locator)
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)

# Leyenda por año
handles = [Line2D([0], [0], color=cmap(i), lw=2, label=str(yy)) for i, yy in enumerate(unique_years)]
ax2.legend(handles=handles, title="Año")

st.pyplot(fig2)