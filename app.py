import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Configuración de la página y Estilo
st.set_page_config(
    page_title="SmartCredit-ML | Scoring de Riesgo",
    page_icon="🏦",
    layout="centered"
)

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
    # 1. Capturamos la edad primero
    age = st.sidebar.slider("Edad", 18, 90, 30)
    income = st.sidebar.number_input("Ingreso Anual (USD)", min_value=0, value=50000)
    
    # 2. Calculamos la antigüedad laboral máxima lógica (Edad - 16)
    # Usamos max(0, ...) para evitar números negativos si el usuario tiene menos de 16
    max_antiguedad = max(0, age - 16)
    
    # Aseguramos que el valor por defecto (5) no sea mayor al máximo permitido
    valor_defecto_antiguedad = min(5, max_antiguedad)
    
    # 3. El slider ahora tiene un límite superior dinámico
    emp_length = st.sidebar.slider("Antigüedad Laboral (años)", 0, max_antiguedad, valor_defecto_antiguedad)
    
    # Datos del préstamo
    loan_amount = st.sidebar.number_input("Monto del Préstamo", min_value=0, value=10000)
    loan_int_rate = st.sidebar.slider("Tasa de Interés (%)", 0.0, 25.0, 11.0)
    
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
        'loan_intent': loan_intent # Aquí se envía en inglés al modelo
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
    # Realizar la predicción usando el Pipeline completo
    prediction = model.predict(input_df)
    
    # CAMBIO APLICADO: Forzamos el tipo float para evitar el error de StreamlitAPIException
    probability = float(model.predict_proba(input_df)[0][1])

    st.divider()
    
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Resultado:")
        if prediction[0] == 0:
            st.success("✅ CRÉDITO APROBADO")
            # --- LÍNEA ELIMINADA PARA MANTENER UN TONO SOBRIO ---
        else:
            st.error("⚠️ RIESGO DE DEFAULT")

    with col2:
        st.write("### Probabilidad de Mora:")
        st.metric(label="Riesgo", value=f"{probability:.2%}")
        
    # Explicación visual del riesgo (Ahora recibe un float nativo de Python)
    st.progress(probability)
    
    if probability > 0.5:
        st.warning("El modelo sugiere que el solicitante tiene un perfil de alto riesgo financiero.")
    else:
        st.info("El solicitante presenta un perfil compatible con las políticas de aprobación.")

    # EXPLICABILIDAD DEL MODELO ---
    st.divider()
    
    # Usamos un expander para no saturar la vista principal
    with st.expander("🔍 Ver análisis de decisión del algoritmo"):
        st.write("¿Qué factores tuvieron más peso para esta predicción?")
        
        # 1. Extraemos el modelo y el preprocesador del Pipeline
        xgb_model = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocessor']
        
        # 2. Obtenemos las importancias y los nombres de las variables transformadas
        importances = xgb_model.feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        
        # 3. Limpiamos los nombres (Scikit-learn les agrega prefijos como 'num__' o 'cat__')
        clean_names = [name.split('__')[-1] for name in feature_names]
        
        # 4. Creamos un DataFrame para graficar
        df_importances = pd.DataFrame({'Importancia': importances}, index=clean_names)
        
        # Ordenamos y tomamos el Top 5 para que el gráfico sea claro
        df_top5 = df_importances.sort_values(by='Importancia', ascending=False).head(5)
        
        # 5. Graficamos usando el componente nativo de Streamlit
        st.bar_chart(df_top5)
        st.caption("El gráfico muestra las 5 variables más determinantes que el modelo XGBoost evaluó para este solicitante en particular.")

# Pie de página profesional
st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por Juan Manuel Landa\nIngeniero en Computación")
