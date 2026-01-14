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

def ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evita ambigüedad label/level: si year/month/day vienen como índice o MultiIndex,
    los baja a columnas sin duplicar nombres.
    """
    idx_names = list(df.index.names) if isinstance(df.index, pd.MultiIndex) else [df.index.name]
    idx_names = [n for n in idx_names if n is not None]

    needs_reset = False
    if isinstance(df.index, pd.MultiIndex):
        needs_reset = True
    elif df.index.name in {"year", "month", "day", "date"}:
        needs_reset = True
    elif any(n in {"year", "month", "day", "date"} for n in idx_names):
        needs_reset = True

    if needs_reset:
        df = df.reset_index()

    # Si por cualquier razón quedaron columnas duplicadas, las compactamos
    # (nos quedamos con la primera ocurrencia)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

st.title("Explorador de caudales — Cotas por estación y año")

DATA_PATH = "data_caudales_diario.pickle"

try:
    df = load_data(DATA_PATH)
    df = ensure_flat_columns(df)
except Exception as e:
    st.error(f"No se pudo cargar el pickle '{DATA_PATH}': {e}")
    st.stop()

# Columnas esperadas
cols = list(df.columns)
required = {"year", "month", "day"}
if not required.issubset(set(cols)):
    st.error("No encuentro columnas year/month/day en el dataset (puede venir con otro esquema).")
    st.write("Columnas detectadas:", cols)
    st.stop()

value_cols = [c for c in cols if c not in ("year", "month", "day")]
if not value_cols:
    st.error("No se encontraron columnas de valores (esperadas aparte de year, month, day).")
    st.write("Columnas detectadas:", cols)
    st.stop()

station = st.selectbox("Estación (columna de valores)", value_cols)

years = sorted(pd.Series(df["year"]).dropna().unique())
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

# Reemplazar ceros por NaN
df_plot[station] = df_plot[station].replace(0, pd.NA)

# Normalizar tipos
df_plot["year"] = pd.to_numeric(df_plot["year"], errors="coerce").astype("Int64")
df_plot["month"] = pd.to_numeric(df_plot["month"], errors="coerce").astype("Int64")
df_plot["day"] = pd.to_numeric(df_plot["day"], errors="coerce").astype("Int64")

# Fecha real para Gráfico 2
df_plot["date"] = pd.to_datetime(
    dict(year=df_plot["year"], month=df_plot["month"], day=df_plot["day"]),
    errors="coerce"
)

# doy auxiliar para Gráfico 1
df_plot["doy"] = df_plot["date"].dt.dayofyear

# Pivot por doy (cada columna un año)
pivot = df_plot.pivot_table(index="doy", columns="year", values=station).sort_index()

# Mapa doy -> (mes, día) con año base (para etiquetas del eje)
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
# Eje X estilo Excel: días + meses debajo
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

# Ticks de días (pocos para no saturar)
dias_mostrar = {1, 8, 15, 22}
day_tick_pos = [i for i, d in enumerate(x_day) if int(d) in dias_mostrar]
day_tick_lab = [str(int(x_day[i])) for i in day_tick_pos]

ax1.set_xticks(day_tick_pos)
ax1.set_xticklabels(day_tick_lab)
ax1.tick_params(axis="x", pad=2)
ax1.set_xlabel("")

# Segunda fila: meses abajo
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
# Una sola línea continua multicolor por año
# Eje X = FECHA real
# =========================
st.subheader("Gráfico 2 — Una sola línea continua (eje X: fecha; colores por año)")

# AQUÍ está el cambio clave: groupby usando Series explícitas (evita ambigüedad)
tmp = df_plot.dropna(subset=["date"]).copy()

df_ts = (
    tmp.groupby([tmp["date"], tmp["year"]], as_index=False)[station]
       .mean()
       .rename(columns={"date": "date", "year": "year"})
       .sort_values("date")
)

years_sorted = sorted(selected_years)
parts = []
for y in years_sorted:
    part = df_ts[df_ts["year"] == y][["date", station, "year"]].sort_values("date")
    parts.append(part)

df_concat = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)

if df_concat.empty:
    st.info("No hay datos para graficar en los años seleccionados.")
    st.stop()

x_dates = df_concat["date"].to_numpy()
y_vals = df_concat[station].to_numpy(dtype=float)
y_year = df_concat["year"].to_numpy(dtype=int)

x_num = mdates.date2num(x_dates)

valid = np.isfinite(y_vals)
points = np.column_stack([x_num, y_vals])

segments = []
segment_years = []
for i in range(len(points) - 1):
    if not (valid[i] and valid[i + 1]):
        continue
    # Evita unir huecos grandes
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
    ax2.plot(df_concat["date"], y_vals, linewidth=2.0)

ax2.set_ylabel(station)
ax2.grid(True)

# Formato de fechas limpio
locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
formatter = mdates.ConciseDateFormatter(locator)
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)

# Leyenda por año
handles = [Line2D([0], [0], color=cmap(i), lw=2, label=str(yy)) for i, yy in enumerate(unique_years)]
ax2.legend(handles=handles, title="Año")

st.pyplot(fig2)