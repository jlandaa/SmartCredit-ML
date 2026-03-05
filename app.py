import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="SmartCredit-ML", layout="centered")

# 2. Carga del modelo entrenado
@st.cache_resource
def load_model():
    # Asegúrate de que el archivo .pkl esté en la misma carpeta que este script
    return joblib.load('smartcredit_model_landa.pkl')

model = load_model()

# 3. Interfaz de usuario
st.title("🏦 Sistema de Scoring de Riesgo Crediticio")
st.markdown("""
Esta aplicación utiliza un modelo **XGBoost** para predecir la probabilidad de incumplimiento de pago (Default) 
basado en el perfil financiero del solicitante.
""")

st.sidebar.header("Datos del Solicitante")

def user_input_features():
    # Entradas numéricas
    age = st.sidebar.slider("Edad", 18, 90, 30)
    income = st.sidebar.number_input("Ingreso Anual (USD)", min_value=0, value=50000)
    emp_length = st.sidebar.slider("Antigüedad Laboral (años)", 0, 50, 5)
    loan_amount = st.sidebar.number_input("Monto del Préstamo", min_value=0, value=10000)
    loan_int_rate = st.sidebar.slider("Tasa de Interés (%)", 0.0, 25.0, 11.0)
    
    # Entradas categóricas (deben coincidir con las opciones del dataset original)
    loan_intent = st.sidebar.selectbox("Motivo del Préstamo", 
        ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    
    # Cálculo automático de variables derivadas si el modelo las requiere
    loan_percent_income = loan_amount / income if income > 0 else 0
    
    data = {
        'person_age': age,
        'person_income': income,
        'person_emp_length': emp_length,
        'loan_amnt': loan_amount,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'loan_intent': loan_intent,
        # Agrega aquí otras columnas si tu modelo las usa (ej. person_home_ownership)
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# 4. Predicción
st.subheader("Análisis de la Solicitud")
if st.button("Evaluar Riesgo"):
    # El pipeline de Scikit-learn se encarga de transformar los datos automáticamente
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    col1, col2 = st.columns(2)

    with col1:
        if prediction[0] == 0:
            st.success("✅ CRÉDITO APROBADO")
        else:
            st.error("⚠️ RIESGO DE DEFAULT DETECTADO")

    with col2:
        prob_default = prediction_proba[0][1]
        st.metric("Probabilidad de Incumplimiento", f"{prob_default:.2%}")

    # Explicación visual rápida
    st.progress(prob_default)
