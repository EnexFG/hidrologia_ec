import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

# ===== Estilo visual (no requiere seaborn instalado) =====
plt.style.use("seaborn-v0_8-whitegrid")

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

# --- Eje X estilo Excel: días (ticks) + meses (segunda fila) ---
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

# leyenda fuera para que no tape líneas
ax1.legend(
    ncols=min(6, max(1, len(selected_years))),
    loc="upper left",
    bbox_to_anchor=(0, 1.02),
    frameon=False
)

st.pyplot(fig1)

st.markdown("---")

# ------------------------
# GRÁFICO 2 (una sola línea continua, segmentos por año con colores distintos)
# ------------------------
unidad = "(msnm)" if station.strip().startswith("Cota") else "(m^3/s)"
st.subheader(f"{station.strip()} {unidad} - Serie Histórica")

years_sorted = sorted(selected_years)
segments = []
segment_years = []

# Preparamos listas para límites de ejes
all_dates_num = []
all_values = []

for y in years_sorted:
    if y not in pivot.columns:
        continue
    
    # Obtenemos los datos del año 'y' y eliminamos nulos
    data_y = pivot[y].dropna()
    
    if data_y.empty:
        continue

    # --- CORRECCIÓN AQUÍ ---
    # Forzamos que el índice (día del año) sea entero. 
    # Si estaba como string u objeto, esto soluciona el error de numpy.
    doy_int = data_y.index.astype(int)
    
    # Construcción de fecha: Año * 1000 + DíaDelAño (ej: 2023001)
    fechas_int = y * 1000 + doy_int
    fechas_str = fechas_int.astype(str)
    
    # Convertimos a datetime
    fechas_dt = pd.to_datetime(fechas_str, format='%Y%j', errors='coerce')
    
    # Convertimos a números de matplotlib
    x_nums = mdates.date2num(fechas_dt)
    y_vals = data_y.values

    # Guardamos para calcular límites globales
    all_dates_num.extend(x_nums)
    all_values.extend(y_vals)

    # Creamos segmentos "punto a punto"
    points = np.column_stack([x_nums, y_vals])
    
    if len(points) > 1:
        # Array de segmentos
        seg = np.stack((points[:-1], points[1:]), axis=1)
        segments.append(seg)
        segment_years.extend([y] * len(seg))

if not segments:
    st.info("No hay datos suficientes para graficar la serie continua.")
    st.stop()

# Concatenamos todos los segmentos
all_segments = np.concatenate(segments)

# Configuración de Colores
unique_years = sorted(list(set(segment_years)))
year_to_idx = {yy: i for i, yy in enumerate(unique_years)}
cvals = np.array([year_to_idx[yy] for yy in segment_years])

cmap = plt.get_cmap("tab10", max(len(unique_years), 1))

lc = LineCollection(all_segments, cmap=cmap)
lc.set_array(cvals)
lc.set_linewidth(1.5)
lc.set_alpha(0.95)

fig2, ax2 = plt.subplots(figsize=(12, 5), constrained_layout=True)
ax2.add_collection(lc)

# --- AJUSTE DE LÍMITES ---
if all_dates_num:
    ax2.set_xlim(min(all_dates_num), max(all_dates_num))
    
    min_y, max_y = min(all_values), max(all_values)
    # Evitar error si min == max
    pad = 0.05 * (max_y - min_y) if max_y != min_y else 1.0
    ax2.set_ylim(min_y - pad, max_y + pad)

# --- FORMATO DE FECHA (AÑOS) ---
if len(unique_years) > 2:
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
else:
    # Si son pocos años, mostramos Mes-Año
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

ax2.set_ylabel(station)
ax2.grid(True, alpha=0.3)

# Leyenda
handles = [Line2D([0], [0], color=cmap(i), lw=2.5, label=str(yy))
           for i, yy in enumerate(unique_years)]
ax2.legend(
    handles=handles,
    loc="upper left",
    bbox_to_anchor=(0, 1.02),
    ncols=min(10, max(1, len(unique_years))),
    frameon=False
)

st.pyplot(fig2)