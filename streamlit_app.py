import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hidrología EC", layout="wide")

meses_map = {
    '01': 'Enero', '02': 'Febrero', '03': 'Marzo', '04': 'Abril',
    '05': 'Mayo', '06': 'Junio', '07': 'Julio', '08': 'Agosto',
    '09': 'Septiembre', '10': 'Octubre', '11': 'Noviembre', '12': 'Diciembre'
}

@st.cache_data
def load_data(path):
    return pd.read_pickle(path)

st.title("Explorador de caudales — Cotas por estación y año")

default_path = 'data_caudales_diario.pickle'

path = st.text_input('Ruta al pickle de datos', value=default_path)

try:
    df = load_data(path)
except Exception as e:
    st.error(f"No se pudo cargar el pickle: {e}")
    st.stop()

# Normalizar nombres de columnas esperadas
cols = list(df.columns)
value_cols = [c for c in cols if c not in ('year', 'month', 'day')]
if not value_cols:
    st.error('No se encontraron columnas de valores (esperadas aparte de year, month, day).')
    st.write('Columnas detectadas:', cols)
    st.stop()

station = st.selectbox('Estación (columna de valores)', value_cols)

years = sorted(df['year'].dropna().unique())
selected_years = st.multiselect('Años a mostrar', years, default=years[-5:] if len(years)>5 else years)

if not selected_years:
    st.info('Selecciona al menos un año.')
    st.stop()

# Filtrar y preparar para graficar
df_plot = df[df['year'].isin(selected_years)][['year','month','day',station]].copy()
# Reemplazar ceros por NaN
df_plot[station].replace(0, pd.NA, inplace=True)

# Crear una columna auxiliar para ordenar los días dentro del año (día del año sin depender del año real)
try:
    md = df_plot['month'].astype(str).str.zfill(2) + '-' + df_plot['day'].astype(str).str.zfill(2)
    df_plot['date_dummy'] = pd.to_datetime(md, format='%m-%d')
    df_plot['doy'] = df_plot['date_dummy'].dt.dayofyear
except Exception:
    # Fallback si month/day no están en formato esperado
    df_plot['doy'] = range(len(df_plot))

pivot = df_plot.pivot_table(index='doy', columns='year', values=station)
# Ordenar por índice (día del año)
pivot = pivot.sort_index()

fig, ax = plt.subplots(figsize=(10, 5))
for y in selected_years:
    if y in pivot.columns:
        ax.plot(pivot.index, pivot[y], label=str(y))

ax.set_xlabel('Día del año (ordenado por mes/día)')
ax.set_ylabel(station)
ax.legend()
ax.grid(True)
st.pyplot(fig)

with st.expander('Ver tabla de datos filtrada'):
    st.dataframe(df_plot.sort_values(['year','doy']).reset_index(drop=True))

st.markdown('---')
st.write('Instrucciones: ejecutar `streamlit run streamlit_app.py` en este directorio.')
