import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
from google.oauth2 import service_account

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

# 1. Configuración de la página y Estilo
st.set_page_config(
    page_title="SmartCredit-ML | Scoring de Riesgo",
    page_icon="🏦",
    layout="centered"
)

# --- CONFIGURACIÓN DE BIGQUERY ---
# Cargamos las credenciales desde los secretos de Streamlit
creds_dict = json.loads(st.secrets["gcp_service_account_json"])
creds = service_account.Credentials.from_service_account_info(creds_dict)

PROJECT_ID = creds_dict["project_id"]
DATASET_ID = "riesgo_crediticio"
TABLE_ID = "historial_predicciones"
FULL_TABLE_ID = f"{DATASET_ID}.{TABLE_ID}"
# ---------------------------------

# 2. Función para cargar el modelo (Cacheada para optimizar rendimiento)
@st.cache_resource
def load_model():
    # El archivo debe estar en la raíz de tu repositorio de GitHub
    return joblib.load('smartcredit_model_landa.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# 3. Interfaz de Usuario
st.title("🏦 SmartCredit-ML")
st.subheader("Sistema Predictivo de Riesgo Crediticio")
st.markdown("""
Esta aplicación utiliza un modelo **XGBoost** entrenado con datos históricos para evaluar la 
probabilidad de incumplimiento de pago (*Default*) de un solicitante.
""")

st.divider()

# 4. Entradas de datos en la barra lateral
st.sidebar.header("📊 Datos del Solicitante")

def get_user_inputs():
    # Datos personales
    age = st.sidebar.slider("Edad", 18, 90, 30)
    
    # Ingreso anual con un mínimo lógico y un TOPE MÁXIMO (Evita números absurdos)
    income = st.sidebar.number_input("Ingreso Anual (USD)", min_value=1000, max_value=5000000, value=50000, step=1000)
    
    # Validación dinámica 1: Antigüedad laboral
    max_antiguedad = max(0, age - 16)
    valor_defecto_antiguedad = min(5, max_antiguedad) 
    emp_length = st.sidebar.slider("Antigüedad Laboral (años)", 0, max_antiguedad, valor_defecto_antiguedad)
    
    # Validación dinámica 2: Capacidad de Endeudamiento
    max_prestamo = max(500, int(income * 5))
    valor_defecto_prestamo = min(10000, max_prestamo) 
    
    # Datos del préstamo ajustados dinámicamente
    loan_amount = st.sidebar.number_input("Monto del Préstamo (USD)", min_value=500, max_value=max_prestamo, value=valor_defecto_prestamo, step=500)
    loan_int_rate = st.sidebar.slider("Tasa de Interés (%)", 1.0, 25.0, 11.0, step=0.1)
    
    # --- Traducción Inglés-Español ---
    traduccion_motivos = {
        'Educación': 'EDUCATION',
        'Gastos Médicos': 'MEDICAL',
        'Emprendimiento (Venture)': 'VENTURE',
        'Uso Personal': 'PERSONAL',
        'Mejoras del Hogar': 'HOMEIMPROVEMENT',
        'Consolidación de Deudas': 'DEBTCONSOLIDATION'
    }
    motivo_espanol = st.sidebar.selectbox("Motivo del Préstamo", list(traduccion_motivos.keys()))
    loan_intent = traduccion_motivos[motivo_espanol]
    # ----------------------------
    
    # Cálculo de feature derivada (loan_percent_income)
    percent_income = loan_amount / income if income > 0 else 0
    
    # Estructura del DataFrame idéntica al entrenamiento
    data = {
        'person_age': age,
        'person_income': income,
        'person_emp_length': emp_length,
        'loan_amnt': loan_amount,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': percent_income,
        'loan_intent': loan_intent 
    }
    return pd.DataFrame([data])

input_df = get_user_inputs()

# 5. Visualización de los datos ingresados
st.write("### Resumen del Perfil")

# Creamos una copia exclusiva para la interfaz visual
df_vista = input_df.copy()

# Renombramos las columnas al español de forma amigable
df_vista.columns = [
    "Edad", 
    "Ingreso Anual (USD)", 
    "Antigüedad Laboral", 
    "Monto del Préstamo", 
    "Tasa de Interés (%)", 
    "% del Ingreso", 
    "Motivo del Préstamo"
]

# Diccionario inverso para traducir el valor de la celda al español
traduccion_inversa = {
    'EDUCATION': 'Educación',
    'MEDICAL': 'Gastos Médicos',
    'VENTURE': 'Emprendimiento (Venture)',
    'PERSONAL': 'Uso Personal',
    'HOMEIMPROVEMENT': 'Mejoras del Hogar',
    'DEBTCONSOLIDATION': 'Consolidación de Deudas'
}

# Aplicamos la traducción solo a esa columna en nuestra copia visual
df_vista["Motivo del Préstamo"] = df_vista["Motivo del Préstamo"].map(traduccion_inversa)

# Mostramos la copia 100% en español, ocultando el índice
st.dataframe(df_vista, hide_index=True)

# 6. Predicción
if st.button("🚀 Evaluar Solicitud"):
    
    # --- 1. VALIDACIÓN LÓGICA DE OUTLIERS ---
    edad_ingresada = input_df['person_age'].iloc[0]
    antiguedad_ingresada = input_df['person_emp_length'].iloc[0]
    
    if antiguedad_ingresada > (edad_ingresada - 16):
        st.error(f"⚠️ Error de validación: Es imposible tener {antiguedad_ingresada} años de antigüedad con {edad_ingresada} años de edad. Verifica los datos.")
        st.stop() 
    # ----------------------------------------

    # Realizar la predicción usando el Pipeline completo
    prediction = model.predict(input_df)
    
    probability = float(model.predict_proba(input_df)[0][1])

    st.divider()
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Resultado:")
        if prediction[0] == 0:
            st.success("✅ CRÉDITO APROBADO")
            decision_texto = "Aprobado"
        else:
            st.error("⚠️ RIESGO DE DEFAULT")
            decision_texto = "Rechazado"

    with col2:
        st.write("### Probabilidad de Mora:")
        st.metric(label="Riesgo Estimado", value=f"{probability:.2%}")
        
    with col3:
        st.write("### Impacto Financiero:")
        # --- CÁLCULO DE VALOR ESPERADO ---
        # ¿Cuánto ganamos si paga? (Intereses)
        ganancia_potencial = loan_amount * (loan_int_rate / 100)
        # ¿Cuánto perdemos si no paga? (Capital prestado)
        perdida_potencial = loan_amount
        
        # Fórmula: (Probabilidad de Pagar * Ganancia) - (Probabilidad de Mora * Pérdida)
        valor_esperado = ((1 - probability) * ganancia_potencial) - (probability * perdida_potencial)
        
        # Mostramos la métrica en color verde si es ganancia, o rojo si es pérdida
        if valor_esperado > 0:
            st.metric(label="Retorno Estadístico Esperado", value=f"${valor_esperado:,.2f} USD", delta="Rentable")
        else:
            st.metric(label="Retorno Estadístico Esperado", value=f"${valor_esperado:,.2f} USD", delta="Pérdida", delta_color="inverse")
        
    st.progress(probability)
    
    if probability > 0.5:
        st.warning("El modelo sugiere que el solicitante tiene un perfil de alto riesgo financiero.")
    else:
        st.info("El solicitante presenta un perfil compatible con las políticas de aprobación.")

    # --- 2. GUARDADO EN BASE DE DATOS (PARA LOOKER) ---
    nuevo_registro = {
        "fecha_evaluacion": datetime.now(),
        "edad": int(edad_ingresada),
        "ingreso_anual": float(input_df['person_income'].iloc[0]),
        "antiguedad_laboral": int(antiguedad_ingresada),
        "monto_prestamo": float(input_df['loan_amnt'].iloc[0]),
        "tasa_interes": float(input_df['loan_int_rate'].iloc[0]),
        "motivo": input_df['loan_intent'].iloc[0],
        "probabilidad_default": probability,
        "decision_final": decision_texto
    }
    
    try:
        # AQUÍ ESTÁ EL CAMBIO: Usamos to_gbq en lugar de to_sql
        df_log = pd.DataFrame([nuevo_registro])
        df_log.to_gbq(
            destination_table=FULL_TABLE_ID, 
            project_id=PROJECT_ID, 
            credentials=creds, 
            if_exists='append'
        )
        st.toast('Registro almacenado en BigQuery para análisis en Looker.', icon='☁️')
    except Exception as e:
        st.sidebar.error(f"Error al guardar en la nube: {e}")
    # --------------------------------------------------

    # EXPLICABILIDAD DEL MODELO ---
    st.divider()
    
    with st.expander("🔍 Ver análisis de decisión del algoritmo"):
        st.write("¿Qué factores tuvieron más peso para esta predicción?")
        
        xgb_model = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocessor']
        
        importances = xgb_model.feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        
        clean_names = [name.split('__')[-1] for name in feature_names]
        
        df_importances = pd.DataFrame({'Importancia': importances}, index=clean_names)
        
        df_top5 = df_importances.sort_values(by='Importancia', ascending=False).head(5)
        
        st.bar_chart(df_top5)
        st.caption("El gráfico muestra las 5 variables más determinantes que el modelo XGBoost evaluó para este solicitante en particular.")

# Pie de página profesional
st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por Juan Manuel Landa\nIngeniero en Computación")


# --- PANEL DE ADMINISTRACIÓN ---
password_admin = st.sidebar.text_input("🔑 Contraseña Admin", type="password")

if password_admin == st.secrets["admin_pass"]:
    st.sidebar.divider()
    st.subheader("☁️ Historial en Google BigQuery")
    try:
        # Leemos la base de datos desde la nube
        query = f"SELECT * FROM `{PROJECT_ID}.{FULL_TABLE_ID}` ORDER BY fecha_evaluacion DESC"
        df_historial = pd.read_gbq(query, project_id=PROJECT_ID, credentials=creds)
        
        st.write(f"**Total de evaluaciones registradas:** {len(df_historial)}")
        st.dataframe(df_historial)
        
        st.download_button(
            label="Descargar datos en CSV",
            data=df_historial.to_csv(index=False).encode('utf-8'),
            file_name='historial_riesgo_bq.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.warning(f"Aún no hay registros o conectando a BigQuery... Detalle: {e}")

    st.divider()
    st.write("### ⚠️ Zona de Peligro")
    
    if st.button("🗑️ Borrar todos los registros de prueba"):
        try:
            from google.cloud import bigquery
            # Nos conectamos con el cliente oficial de BigQuery
            client = bigquery.Client(credentials=creds, project=PROJECT_ID)
            
            #Como el tier gratuito no permite DML (DELETE), 
            #eliminamos la tabla completa. Pandas la recreará sola en la próxima predicción.
            client.delete_table(FULL_TABLE_ID, not_found_ok=True) 
            
            st.success("✅ Registros eliminados de BigQuery exitosamente (Tabla reiniciada).")
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo vaciar la tabla: {e}")
