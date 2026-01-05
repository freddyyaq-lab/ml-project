import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# 1. Configuración de la Página
# ==========================================
st.set_page_config(
    page_title="Evaluador de Riesgo de Crédito",
    page_icon="💰",
    layout="centered"
)

# ==========================================
# 2. Función para cargar el modelo
# ==========================================
@st.cache_resource
def load_model():
    # Intentamos buscar el modelo en diferentes rutas comunes
    paths = ["src/best_model.pkl", "best_model.pkl", "../src/best_model.pkl"]
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

model = load_model()

# ==========================================
# 3. Interfaz de Usuario
# ==========================================
st.title("💰 Sistema de Evaluación de Crédito")
st.markdown("""
Esta herramienta utiliza Inteligencia Artificial para predecir la probabilidad de que un cliente pague su crédito a tiempo.
""")

if model is None:
    st.error("🚨 No se encontró el archivo 'best_model.pkl'. Por favor, ejecuta primero el script de entrenamiento.")
else:
    st.subheader("📝 Datos del Solicitante")
    
    with st.form("credit_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            tipo_credito = st.selectbox("Tipo de Crédito", ["1", "2", "3", "4", "5", "6", "7", "8", "9"])
            capital_prestado = st.number_input("Monto del Préstamo ($)", min_value=0.0, value=1000000.0)
            plazo_meses = st.number_input("Plazo (Meses)", min_value=1, value=12)
            edad_cliente = st.number_input("Edad del Cliente", min_value=18, max_value=100, value=30)
            tipo_laboral = st.selectbox("Ocupación", ["Empleado", "Independiente", "Pensionado", "Otro"])
            salario_cliente = st.number_input("Salario Mensual ($)", min_value=0.0, value=2000000.0)
            total_otros_prestamos = st.number_input("Total Otros Préstamos ($)", min_value=0.0, value=0.0)
            cuota_pactada = st.number_input("Cuota Mensual Pactada ($)", min_value=0.0, value=150000.0)

        with col2:
            puntaje_datacredito = st.number_input("Score Datacrédito", min_value=0, max_value=990, value=700)
            cant_creditosvigentes = st.number_input("Créditos Vigentes", min_value=0, value=1)
            huella_consulta = st.number_input("Huellas de Consulta (Último mes)", min_value=0, value=0)
            creditos_sectorFinanciero = st.number_input("Créditos Sector Financiero", min_value=0, value=0)
            creditos_sectorCooperativo = st.number_input("Créditos Sector Cooperativo", min_value=0, value=0)
            creditos_sectorReal = st.number_input("Créditos Sector Real", min_value=0, value=0)
            promedio_ingresos_datacredito = st.number_input("Ingresos Promedio (Buró) ($)", min_value=0.0, value=2000000.0)
            tendencia_ingresos = st.selectbox("Tendencia de Ingresos", ["Creciente", "Estable", "Decreciente"])

        submit = st.form_submit_button("🚀 Evaluar Riesgo")

    # ==========================================
    # 4. Lógica de Predicción
    # ==========================================
    if submit:
        # Crear un DataFrame con los nombres exactos de las columnas usadas en el entrenamiento
        input_df = pd.DataFrame([{
            'tipo_credito': str(tipo_credito),
            'capital_prestado': capital_prestado,
            'plazo_meses': plazo_meses,
            'edad_cliente': edad_cliente,
            'tipo_laboral': str(tipo_laboral),
            'salario_cliente': salario_cliente,
            'total_otros_prestamos': total_otros_prestamos,
            'cuota_pactada': cuota_pactada,
            'puntaje_datacredito': puntaje_datacredito,
            'cant_creditosvigentes': cant_creditosvigentes,
            'huella_consulta': huella_consulta,
            'creditos_sectorFinanciero': creditos_sectorFinanciero,
            'creditos_sectorCooperativo': creditos_sectorCooperativo,
            'creditos_sectorReal': creditos_sectorReal,
            'promedio_ingresos_datacredito': promedio_ingresos_datacredito,
            'tendencia_ingresos': str(tendencia_ingresos)
        }])

        try:
            # Obtener predicción y probabilidad
            probabilidad = model.predict_proba(input_df)[0][1]
            resultado = model.predict(input_df)[0]

            st.markdown("---")
            st.subheader("📊 Resultado del Análisis")

            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if resultado == 1:
                    st.success("✅ CRÉDITO APROBADO")
                else:
                    st.error("❌ CRÉDITO RECHAZADO")

            with col_res2:
                st.metric("Confianza de Pago", f"{probabilidad:.1%}")

            # Barra de progreso visual
            st.progress(float(probabilidad))
            
            if probabilidad < 0.5:
                st.warning("El cliente presenta un perfil de riesgo elevado para el cumplimiento de pagos.")
            else:
                st.info("El cliente tiene una probabilidad sólida de cumplir con sus obligaciones.")

        except Exception as e:
            st.error(f"Error al procesar la predicción: {e}")