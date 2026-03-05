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
    # Datos personales
    age = st.sidebar.slider("Edad", 18, 90, 30)
    income = st.sidebar.number_input("Ingreso Anual (USD)", min_value=0, value=50000)
    emp_length = st.sidebar.slider("Antigüedad Laboral (años)", 0, 50, 5)
    
    # Datos del préstamo
    loan_amount = st.sidebar.number_input("Monto del Préstamo", min_value=0, value=10000)
    loan_int_rate = st.sidebar.slider("Tasa de Interés (%)", 0.0, 25.0, 11.0)
    
    # Motivo del préstamo (Debe coincidir con las categorías del entrenamiento)
    loan_intent = st.sidebar.selectbox("Motivo del Préstamo", 
        ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    
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
st.dataframe(input_df, hide_index=True)

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
            st.balloons()
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

# Pie de página profesional
st.sidebar.markdown("---")
st.sidebar.info(f"Desarrollado por Juan Manuel Landa\nIngeniero en Computación")
