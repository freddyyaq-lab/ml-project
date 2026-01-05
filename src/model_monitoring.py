import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

try:
    from Cargar_datos import Cargar_datos
    from ft_engineering import FeatureEngineering
except ImportError:

    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from src.Cargar_datos import Cargar_datos
    from src.ft_engineering import FeatureEngineering

st.set_page_config(page_title="Tablero de Monitoreo", layout="wide", page_icon="📈")

DATASET_PATH = "./Base_de_datos.xlsx"  # Ajusta según tu estructura
MODEL_PATHS = ["src/best_model.pkl", "best_model.pkl", "../src/best_model.pkl"]
LOG_FILE = "src/monitoring_log.csv"

@st.cache_resource
def load_resources():

    model = None
    for path in MODEL_PATHS:
        try:
            model = joblib.load(path)
            break
        except:
            continue
            
    if model is None:
        return None, None

    try:
        possible_data_paths = ["../Base_de_datos.xlsx", "Base_de_datos.xlsx"]
        df = None
        for dp in possible_data_paths:
            if os.path.exists(dp):
                loader = Cargar_datos(dp)
                df = loader.carga_datos()
                break
        
        if df is None:
            return None, None

        fe = FeatureEngineering(df, target_col='Pago_atiempo')
        fe._drop_manual_columns()
        fe._remove_highly_correlated_features(threshold=0.85)
        reference_data = fe.df.copy()
        
        return model, reference_data
    except Exception as e:
        return None, None

model, reference_data = load_resources()


def simulate_new_predictions(n_samples=50):
    if reference_data is None: return None
    
    new_data = reference_data.sample(n=n_samples, replace=True).copy()

    new_data['salario_cliente'] = new_data['salario_cliente'] * np.random.uniform(1.05, 1.20, size=len(new_data))
    new_data['capital_prestado'] = new_data['capital_prestado'] * 1.10
    
    cat_cols = new_data.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        new_data[col] = new_data[col].astype(str)
        
    preds = model.predict(new_data)
    new_data['prediction'] = preds
    new_data['timestamp'] = pd.Timestamp.now()
    
    return new_data

st.title("📈 Dashboard de Monitoreo (Producción)")

if model is None:
    st.error("Error: No se encontró el modelo ('best_model.pkl') o la base de datos.")
    st.stop()

current_data = pd.DataFrame()

if os.path.exists(LOG_FILE):
    try:
        current_data = pd.read_csv(LOG_FILE)
    except:
        current_data = pd.DataFrame()

# --- BARRA LATERAL ---
st.sidebar.header("⚙️ Panel de Control")

if st.sidebar.button("🔄 Simular llegada de clientes"):
    with st.spinner("Procesando nuevos datos..."):
        new_batch = simulate_new_predictions(n_samples=50)
        
        if new_batch is not None:
            # Guardar en CSV
            if not current_data.empty:
                new_batch.to_csv(LOG_FILE, mode='a', header=False, index=False)
            else:
                new_batch.to_csv(LOG_FILE, index=False)
            
            # Recargar datos inmediatamente para actualizar gráficas
            current_data = pd.concat([current_data, new_batch], ignore_index=True)
            st.sidebar.success(f"✅ Se procesaron {len(new_batch)} solicitudes.")
            st.rerun() # Recarga la página para mostrar los datos nuevos

if st.sidebar.button("🗑️ Reiniciar Historial"):
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        current_data = pd.DataFrame() # Limpiamos la variable en memoria
        st.sidebar.warning("Historial eliminado.")
        st.rerun()

# --- VISUALIZACIÓN DEL DASHBOARD ---
# Aquí verificamos si hay datos ANTES de intentar graficar
if current_data.empty:
    st.info("👋 **Bienvenido al sistema de monitoreo.**")
    st.markdown("No hay datos registrados todavía.")
    st.markdown("👉 **Presiona el botón 'Simular llegada de clientes' en la barra lateral para generar datos de prueba.**")
    
else:
    # SI HAY DATOS, MOSTRAR TABS
    tab1, tab2, tab3 = st.tabs(["📊 Desempeño", "🚨 Detección de Cambios (Drift)", "📂 Datos Crudos"])

    with tab1:
        st.subheader("Métricas de Operación")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Solicitudes", len(current_data))
        with col2: st.metric("Tasa Aprobación", f"{(current_data['prediction'] == 1).mean():.1%}")
        with col3: st.metric("Monto Promedio", f"${current_data['capital_prestado'].mean():,.0f}")
        with col4: st.metric("Salario Promedio", f"${current_data['salario_cliente'].mean():,.0f}")
        
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            counts = current_data['prediction'].value_counts().reset_index()
            counts.columns = ['Resultado', 'Cantidad']
            counts['Resultado'] = counts['Resultado'].map({1: 'Aprobado', 0: 'Rechazado'})
            fig_pie = px.pie(counts, values='Cantidad', names='Resultado', title="Distribución de Decisiones", 
                             color='Resultado', color_discrete_map={'Aprobado':'green', 'Rechazado':'red'})
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_hist = px.histogram(current_data, x="salario_cliente", title="Distribución de Salarios Actuales")
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.subheader("Comparativo: Entrenamiento vs. Producción")
        col_drift1, col_drift2 = st.columns(2)
        with col_drift1:
            fig_box1 = go.Figure()
            fig_box1.add_trace(go.Box(y=reference_data['salario_cliente'], name='Entrenamiento', marker_color='blue'))
            fig_box1.add_trace(go.Box(y=current_data['salario_cliente'], name='Producción', marker_color='orange'))
            fig_box1.update_layout(title="Comparación: Salarios")
            st.plotly_chart(fig_box1, use_container_width=True)
        with col_drift2:
            fig_box2 = go.Figure()
            fig_box2.add_trace(go.Box(y=reference_data['capital_prestado'], name='Entrenamiento', marker_color='blue'))
            fig_box2.add_trace(go.Box(y=current_data['capital_prestado'], name='Producción', marker_color='orange'))
            fig_box2.update_layout(title="Comparación: Monto Solicitado")
            st.plotly_chart(fig_box2, use_container_width=True)

    with tab3:
        st.dataframe(current_data.tail(100))
        st.download_button("Descargar Reporte CSV", current_data.to_csv(index=False), "reporte.csv", "text/csv")