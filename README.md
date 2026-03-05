# 🏦 SmartCredit-ML: Predictive Credit Risk Scoring

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jml-smartcredit-ml.streamlit.app/)

## 📌 Descripción del Proyecto
SmartCredit-ML es un sistema predictivo de machine learning diseñado para evaluar el riesgo crediticio de solicitantes de préstamos. La aplicación automatiza el proceso de *scoring*, prediciendo la probabilidad de incumplimiento de pago (Default) basándose en el perfil financiero y demográfico del usuario.

Este proyecto demuestra un ciclo completo de datos: desde la limpieza y transformación (Data Wrangling), pasando por el entrenamiento del modelo, hasta el despliegue de una aplicación web interactiva.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.12+
* **Machine Learning:** XGBoost, Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Despliegue & UI:** Streamlit, Streamlit Community Cloud
* **Serialización:** Joblib

## 🧠 Arquitectura del Modelo
El núcleo del sistema es un **Pipeline de Scikit-learn** que garantiza la robustez del preprocesamiento en producción, evitando la filtración de datos (data leakage):
1. **Preprocesamiento:** Imputación de valores nulos mediante la mediana (`SimpleImputer`), escalado de variables numéricas (`StandardScaler`) y codificación de variables categóricas (`OneHotEncoder`).
2. **Modelo Predictivo:** Clasificador **XGBoost**.
3. **Manejo de Desbalance:** Se implementó el parámetro `scale_pos_weight` en XGBoost para penalizar fuertemente los falsos negativos, priorizando la detección de perfiles de alto riesgo en un dataset donde los "buenos pagadores" son mayoría.

## 🚀 Cómo ejecutarlo localmente
1. Clona el repositorio:
```bash
git clone [https://github.com/tu-usuario/smartcredit-ml.git](https://github.com/tu-usuario/smartcredit-ml.git)
```
2. Instala las dependencias:
```bash
pip install -r requirements.txt
```
3. Ejecuta la aplicación de Streamlit:
```bash
streamlit run app.py
```

## 👨‍💻 Autor
## Juan Manuel Landa - Ingeniero en Computación
