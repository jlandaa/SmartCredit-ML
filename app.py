import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
from google.oauth2 import service_account
import shap 
import matplotlib.pyplot as plt 

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
modo_prueba = st.sidebar.toggle("Modo Simulación", value=True, help="Si está activo, no se guardarán datos en BigQuery")
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
# --- 6.1 VALIDACIONES Y ALERTA MLOPS (OOD) ---
if st.button("🚀 Evaluar Solicitud"):
    
    # --- 1. VALIDACIÓN LÓGICA DE OUTLIERS (Reglas duras) ---
    edad_ingresada = input_df['person_age'].iloc[0]
    antiguedad_ingresada = input_df['person_emp_length'].iloc[0]
    
    if antiguedad_ingresada > (edad_ingresada - 16):
        st.error(f"⚠️ Error de validación: Es imposible tener {antiguedad_ingresada} años de antigüedad con {edad_ingresada} años de edad. Verifica los datos.")
        st.stop() 
    # ----------------------------------------

    # --- ALERTA MLOPS: OUT-OF-DISTRIBUTION (OOD) ---
    # Definimos los percentiles 99 aproximados de nuestra base histórica
    umbrales_ood = {
        'ingreso': 150000,      # El 99% gana menos de 150k
        'monto': 25000,         # El 99% pide menos de 25k
        'tasa': 20.0,           # Rara vez damos tasas mayores al 20%
        'antiguedad': 20        # Rara vez alguien tiene más de 20 años en el mismo empleo
    }
    
    alertas_ood = []
    if input_df['person_income'].iloc[0] > umbrales_ood['ingreso']:
        alertas_ood.append("Ingreso excepcionalmente alto")
    if input_df['loan_amnt'].iloc[0] > umbrales_ood['monto']:
        alertas_ood.append("Monto de préstamo inusual")
    if input_df['loan_int_rate'].iloc[0] > umbrales_ood['tasa']:
        alertas_ood.append("Tasa de interés extrema")
    if input_df['person_emp_length'].iloc[0] > umbrales_ood['antiguedad']:
        alertas_ood.append("Antigüedad laboral atípica")
        
    if alertas_ood:
        st.warning(f"🛡️ **Alerta de MLOps (Data Drift):** Este perfil tiene características en el 1% superior de nuestra base histórica ({', '.join(alertas_ood)}). El modelo está operando en terreno desconocido y la predicción podría tener un margen de error mayor. Se recomienda revisión manual.")
    # ----------------------------------------
    # --- 6.2 CÁLCULO Y RESULTADOS (XGBOOST) ---
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
        
        # --- CORRECCIÓN: Extraemos los valores del DataFrame ---
        monto_prestamo_calc = float(input_df['loan_amnt'].iloc[0])
        tasa_interes_calc = float(input_df['loan_int_rate'].iloc[0])
        
        # --- CÁLCULO DE VALOR ESPERADO ---
        # ¿Cuánto ganamos si paga? (Intereses)
        ganancia_potencial = monto_prestamo_calc * (tasa_interes_calc / 100)
        # ¿Cuánto perdemos si no paga? (Capital prestado)
        perdida_potencial = monto_prestamo_calc
        
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

    # --- 6.3 ASISTENTE NARRATIVO (NLG) ---
    st.divider()
    st.write("### 🤖 Asistente Narrativo")
    
    def generar_reporte_narrativo(df, prob, valor_esp):
        # 1. Definir el nivel de riesgo
        riesgo_str = "Bajo" if prob < 0.2 else "Moderado" if prob < 0.5 else "Alto"

        # 2. Buscar la principal fortaleza (Reglas de negocio)
        if df['person_emp_length'].iloc[0] >= 5:
            fortaleza = "su alta estabilidad laboral"
        elif df['person_income'].iloc[0] > 60000:
            fortaleza = "su sólido nivel de ingresos"
        else:
            fortaleza = "el perfil general de los datos ingresados"

        # 3. Buscar la principal debilidad (Reglas de negocio)
        if df['loan_percent_income'].iloc[0] > 0.3:
            alerta = "su alto nivel de endeudamiento en relación a su salario"
        elif df['loan_int_rate'].iloc[0] > 15:
            alerta = "la alta tasa de interés pactada para esta operación"
        else:
            alerta = "el riesgo estadístico base del segmento"

        # 4. Redacción natural
        reporte = f"El modelo indica un perfil de riesgo **{riesgo_str}** (Probabilidad de mora: {prob:.1%}). "
        reporte += f"Su principal fortaleza para la evaluación crediticia es {fortaleza}. "
        reporte += f"Sin embargo, el factor de mayor precaución detectado es {alerta}, lo que resulta en un impacto financiero esperado de **${valor_esp:,.2f} USD**."

        return reporte

    # Generamos el texto y lo mostramos en una caja azul profesional
    reporte_texto = generar_reporte_narrativo(input_df, probability, valor_esperado)
    st.info(f"📝 **Resumen Ejecutivo:** {reporte_texto}")
    # --------------------------------------------------

    # --- 6.4 GUARDADO EN BASE DE DATOS (BIGQUERY) ---
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
    if not modo_prueba: # Solo guardamos si el modo prueba está APAGADO
        try:
            # Usamos to_gbq en lugar de to_sql
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
    else:
        # Si el modo prueba está ENCENDIDO, mostramos este cartel en vez de guardar
        st.info("💡 Modo Simulación activo: El resultado se calculó, pero NO se guardó en la base de datos para no ensuciar las métricas de Looker.")
    # --------------------------------------------------

    # --- 6.5 EXPLICABILIDAD DEL MODELO (SHAP) ---
    st.divider()
    
    with st.expander("🔍 Análisis profundo: ¿Por qué se tomó esta decisión? (SHAP)"):
        st.write("Este gráfico de cascada (Waterfall) desglosa matemáticamente cómo cada dato del solicitante empujó la decisión hacia la aprobación o el rechazo.")
        
        # 1. Extraemos los componentes del Pipeline
        xgb_model = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocessor']
        
        # 2. Transformamos el dato del usuario tal como lo hace el modelo internamente
        input_processed = preprocessor.transform(input_df)
        
        # 3. Limpiamos los nombres de las columnas para que el gráfico sea legible
        feature_names = preprocessor.get_feature_names_out()
        clean_names = [name.split('__')[-1] for name in feature_names]
        
        # 4. Calculamos los valores SHAP (Usamos TreeExplainer, optimizado para XGBoost)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(input_processed)
        
        # Le inyectamos los nombres limpios al objeto SHAP
        shap_values.feature_names = clean_names
        
        # 5. Generamos el gráfico en Matplotlib para pasarlo a Streamlit
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Usamos shap_values[0] porque estamos evaluando a 1 solo solicitante
        shap.plots.waterfall(shap_values[0], max_display=7, show=False)
        
        # Renderizamos en Streamlit y limpiamos la memoria
        st.pyplot(fig)
        plt.clf() 
        
        st.caption("🔵 Valores azules: Disminuyen el riesgo de mora. | 🔴 Valores rojos: Aumentan el riesgo de mora.")

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
