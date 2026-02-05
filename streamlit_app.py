import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import os

plt.style.use("seaborn-v0_8-whitegrid")

st.set_page_config(page_title="Hidrología EC", layout="wide")

meses_abrev = {
    '01': 'Ene', '02': 'Feb', '03': 'Mar', '04': 'Abr',
    '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Ago',
    '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dic'
}

@st.cache_data
def load_data(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_pickle(path)

st.title("Comparativo multianual de niveles y caudales — Fuente: CELEC")

DATA_PATH = "data_caudales_diario.pickle"

try:
    mtime = os.path.getmtime(DATA_PATH)
    df = load_data(DATA_PATH, mtime)
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

default_station = "CotaMaz" 

default_index = value_cols.index(default_station) if default_station in value_cols else 0
station = st.selectbox("Cota/Caudal:", value_cols, index=default_index)

years = sorted(df["year"].dropna().unique())

N = 3
default_years = years[-N:] if len(years) >= N else years

selected_years = st.multiselect("Años a mostrar:", years, default=default_years)

if not selected_years:
    st.info("Selecciona al menos un año.")
    st.stop()

# ------------------------
# DASHBOARD
# ------------------------
st.markdown("---")
st.subheader("Dashboard — Resumen")
st.caption(
    "Supuestos: dashboard genérico por estación seleccionada; KPIs calculados con datos diarios "
    "de los años elegidos; tendencias agregadas mensualmente; tabla de desglose por mes y año. "
    "Reemplaza o amplía métricas según necesidad."
)

df_dash = df[df["year"].isin(selected_years)][["year", "month", "day", station]].copy()
df_dash[station] = df_dash[station].replace(0, pd.NA)

try:
    df_dash["fecha"] = pd.to_datetime(df_dash[["year", "month", "day"]])
except Exception as e:
    st.error(f"Error creando fechas para dashboard: {e}. Verifica year/month/day.")
    st.stop()

df_dash = df_dash.dropna(subset=[station]).sort_values("fecha")

if df_dash.empty:
    st.info("No hay datos disponibles para el dashboard con la selección actual.")
    st.stop()

unidad = "(msnm)" if station.strip().startswith("Cota") else "(m^3/s)"

latest_row = df_dash.iloc[-1]
latest_value = latest_row[station]
latest_date = latest_row["fecha"].date()

window = 7
last_window = df_dash.tail(window)[station].mean() if len(df_dash) >= window else df_dash[station].mean()
prev_window = (
    df_dash.iloc[-(2 * window):-window][station].mean()
    if len(df_dash) >= 2 * window
    else np.nan
)
delta_window = last_window - prev_window if pd.notna(prev_window) else np.nan

min_value = df_dash[station].min()
max_value = df_dash[station].max()
mean_value = df_dash[station].mean()

kpi_cols = st.columns(4)
kpi_cols[0].metric("Último valor", f"{latest_value:.2f} {unidad}", f"{latest_date}")
kpi_cols[1].metric("Promedio", f"{mean_value:.2f} {unidad}")
kpi_cols[2].metric("Mín / Máx", f"{min_value:.2f} / {max_value:.2f} {unidad}")
if pd.notna(delta_window):
    kpi_cols[3].metric(f"Cambio {window}d", f"{last_window:.2f} {unidad}", f"{delta_window:+.2f}")
else:
    kpi_cols[3].metric(f"Cambio {window}d", f"{last_window:.2f} {unidad}", "N/D")

st.markdown("### Tendencias")
df_month = (
    df_dash
    .assign(month_start=lambda d: d["fecha"].dt.to_period("M").dt.to_timestamp())
    .groupby("month_start")[station]
    .mean()
    .reset_index()
)

fig_trend, ax_trend = plt.subplots(figsize=(12, 4), constrained_layout=True)
ax_trend.plot(df_month["month_start"], df_month[station], linewidth=2.2)
ax_trend.set_ylabel(f"{station} {unidad}")
ax_trend.grid(True, alpha=0.3)
ax_trend.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_trend.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
st.pyplot(fig_trend)

st.markdown("### Desglose mensual por año")
df_table = (
    df_dash
    .assign(month=lambda d: d["fecha"].dt.month)
    .pivot_table(index="month", columns="year", values=station, aggfunc="mean")
    .sort_index()
)
df_table.index = df_table.index.map(lambda m: meses_abrev[str(m).zfill(2)])
st.dataframe(df_table.style.format("{:.2f}"), use_container_width=True)

st.markdown("---")

df_plot = df[df["year"].isin(selected_years)][["year", "month", "day", station]].copy()
df_plot[station] = df_plot[station].replace(0, pd.NA)

df_plot["month"] = df_plot["month"].astype(int).astype(str).str.zfill(2)
df_plot["day"] = df_plot["day"].astype(int).astype(str).str.zfill(2)

df_plot["month_name"] = df_plot["month"].map(meses_abrev)
df_plot["day_int"] = df_plot["day"].astype(int)

md = df_plot["month"] + "-" + df_plot["day"]
df_plot["date_dummy"] = pd.to_datetime(md, format="%m-%d", errors="coerce")
df_plot["doy"] = df_plot["date_dummy"].dt.dayofyear

pivot = df_plot.pivot_table(index="doy", columns="year", values=station).sort_index()


doy_to_md = (
    df_plot.dropna(subset=["doy"])
           .drop_duplicates(subset=["doy"])
           .set_index("doy")[["month", "day"]]
           .sort_index()
)

doy_index = pivot.index.astype(int)
md_for_doy = doy_to_md.reindex(doy_index)

missing = md_for_doy["month"].isna()
if missing.any():
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

# ------------------------
# GRÁFICO 1
# ------------------------
unidad = "(msnm)" if station.strip().startswith("Cota") else "(m^3/s)"
st.subheader(f"{station.strip()} {unidad}")

fig1, ax1 = plt.subplots(figsize=(12, 5), constrained_layout=True)
xpos = np.arange(len(pivot.index))

# líneas un poco más legibles
for y in selected_years:
    if y in pivot.columns:
        ax1.plot(
            xpos, pivot[y].values,
            label=str(y),
            linewidth=2.0,
            alpha=0.9
        )

ax1.set_ylabel(station)
ax1.grid(True, alpha=0.3)

dias_mostrar = {1, 10, 22}
day_tick_pos = [i for i, d in enumerate(x_day) if int(d) in dias_mostrar]
day_tick_lab = [str(int(x_day[i])) for i in day_tick_pos]

ax1.set_xticks(day_tick_pos)
ax1.set_xticklabels(day_tick_lab)
ax1.tick_params(axis="x", pad=2)

secax = ax1.secondary_xaxis('bottom')
secax.spines['bottom'].set_position(('outward', 22))

month_start_pos = []
month_labels = []
seen = set()
for i, m in enumerate(x_month):
    if m not in seen:
        seen.add(m)
        month_start_pos.append(i)
        month_labels.append(meses_abrev[m])

secax.set_xticks(month_start_pos)
secax.set_xticklabels(month_labels)

ax1.set_xlabel("")
secax.set_xlabel("")

ax1.legend(
    ncols=min(6, max(1, len(selected_years))),
    loc="upper left",
    bbox_to_anchor=(0, 1.02),
    frameon=False
)

st.pyplot(fig1)

st.markdown("---")

# ------------------------
# GRÁFICO 2
# ------------------------
unidad = "(msnm)" if station.strip().startswith("Cota") else "(m^3/s)"
st.subheader(f"{station.strip()} {unidad} - Serie Histórica")


mask_years = df["year"].isin(selected_years)
df_ts = df[mask_years].copy()


try:
    df_ts["fecha"] = pd.to_datetime(df_ts[["year", "month", "day"]])
except Exception as e:
    st.error(f"Error creando fechas: {e}. Verifica que las columnas year/month/day sean numéricas.")
    st.stop()

df_ts = df_ts.sort_values("fecha")
df_ts[station] = df_ts[station].replace(0, pd.NA)
df_ts = df_ts.dropna(subset=[station])

if df_ts.empty:
    st.info("No hay datos para graficar en los años seleccionados.")
    st.stop()


x_nums = mdates.date2num(df_ts["fecha"])
y_vals = df_ts[station].values
years_vals = df_ts["year"].values

points = np.column_stack([x_nums, y_vals])

segments = np.stack((points[:-1], points[1:]), axis=1)


unique_years_ts = sorted(df_ts["year"].unique())
year_to_idx = {yy: i for i, yy in enumerate(unique_years_ts)}

cvals = np.array([year_to_idx[yy] for yy in years_vals[:-1]])

fig2, ax2 = plt.subplots(figsize=(12, 5), constrained_layout=True)

cmap = plt.get_cmap("tab10", max(len(unique_years_ts), 1))
lc = LineCollection(segments, cmap=cmap)
lc.set_array(cvals)
lc.set_linewidth(1.5)
lc.set_alpha(0.95)

ax2.add_collection(lc)

ax2.set_xlim(x_nums.min(), x_nums.max())
pad = 0.05 * (y_vals.max() - y_vals.min()) if y_vals.max() != y_vals.min() else 1.0
ax2.set_ylim(y_vals.min() - pad, y_vals.max() + pad)

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)

ax2.set_ylabel(station)
ax2.grid(True, alpha=0.3)

handles = [Line2D([0], [0], color=cmap(i), lw=2.5, label=str(yy))
           for i, yy in enumerate(unique_years_ts)]
ax2.legend(
    handles=handles,
    loc="upper left",
    bbox_to_anchor=(0, 1.02),
    ncols=min(10, max(1, len(unique_years_ts))),
    frameon=False
)

st.pyplot(fig2)
