# 🏦 SmartCredit-ML: Predictive Credit Risk Scoring
![Status: Maintained](https://img.shields.io/badge/Status-Maintained-brightgreen?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jml-smartcredit-ml.streamlit.app/)

## 📌 Descripción del Proyecto
SmartCredit-ML es una solución integral de Scoring de Riesgo Crediticio que transforma probabilidades estadísticas en decisiones de negocio claras. A diferencia de un modelo predictivo tradicional, esta plataforma automatiza el proceso de evaluación incorporando capas de explicabilidad algorítmica (XAI), monitoreo preventivo de datos (MLOps) y generación de lenguaje natural (NLG).

El sistema no solo predice la probabilidad de incumplimiento de pago (Default) basándose en el perfil financiero y demográfico del usuario, sino que calcula el impacto financiero esperado en dólares y traduce la matemática compleja en un resumen ejecutivo legible.

Este proyecto demuestra un ciclo de vida de datos End-to-End de grado empresarial: desde la limpieza y el entrenamiento del modelo (XGBoost), pasando por la detección de Data Drift y la persistencia en la nube (Google BigQuery), hasta el despliegue de una aplicación web interactiva diseñada para la toma de decisiones gerenciales.

## ✨ Características Destacadas (Enterprise Architecture)
Este proyecto implementa prácticas avanzadas de **MLOps** y **Data Engineering** para entornos de producción:

* **MLOps (Data Drift Detection):** Incorpora un sistema de monitoreo *Out-of-Distribution* (OOD). Si un usuario ingresa datos que se desvían del 1% superior de la distribución histórica, la app dispara una alerta de riesgo por operar en terreno desconocido para el modelo.
* **Explainable AI (XAI) con SHAP:** Rompe la "caja negra" del algoritmo mediante gráficos de cascada (*Waterfall plots*). Permite visualizar exactamente cuánto sumó o restó cada variable (edad, ingresos, tasa) a la probabilidad final. 
* **Asistente Narrativo (NLG Determinista):** Traduce los resultados técnicos a lenguaje humano. Genera un **Resumen Ejecutivo** automático que identifica la principal fortaleza y debilidad del solicitante sin necesidad de interpretar gráficos. 
* **Análisis de Valor Esperado:** Calcula el impacto financiero real en USD. Cruza la probabilidad de mora con la tasa de interés para determinar si el préstamo es estadísticamente rentable para la institución. 
* **Cloud Data Pipeline:** Persistencia automática de evaluaciones en **Google BigQuery**, permitiendo la trazabilidad completa del modelo. Incluye un **Modo Simulación** para testear la herramienta sin ensuciar las métricas productivas. 

## 🛠️ Stack Tecnológico
* **Machine Learning:** Python 3.12+, XGBoost, Scikit-learn, **SHAP** (Interpretabilidad). 
* **Data Engineering:** **Google BigQuery**, Pandas, NumPy. 
* **Visualización & BI:** **Looker Studio**, Matplotlib, Streamlit. 
* **Despliegue:** Streamlit Community Cloud. 

## 📊 Dashboard Gerencial
Las predicciones realizadas por el modelo se centralizan en un tablero de control para el monitoreo de KPIs de riesgo:

👉 [Ver Dashboard en Looker Studio](https://lookerstudio.google.com/reporting/c48718c7-f0a5-4672-82bf-bdd38e346c76/page/BenrF) 

## 🧠 Arquitectura del Sistema
El núcleo de la aplicación está diseñado como un **Pipeline *End-to-End*** que abarca desde la validación del dato crudo hasta la explicabilidad algorítmica y la persistencia en la nube:

1. **Capa de Validación y Monitoreo (MLOps):** Implementación de reglas de negocio duras y un sistema de detección de anomalías estadísticas (*Data Drift / Out-of-Distribution*) que audita el perfil del usuario antes de la inferencia.
2. **Pipeline de Preprocesamiento:** Construido nativamente con Scikit-learn para evitar la filtración de datos (*Data Leakage*) en producción. Incluye imputación de nulos (`SimpleImputer`), escalado numérico (`StandardScaler`) y codificación categórica (`OneHotEncoder`).
3. **Motor Predictivo (XGBoost):** Clasificador de alto rendimiento optimizado con el parámetro `scale_pos_weight` para penalizar drásticamente los falsos negativos (priorizando la detección del riesgo en un dataset altamente desbalanceado).
4. **Capa de Explicabilidad (XAI & NLG):**
   * **Auditoría Matemática:** Integración con **SHAP** (`TreeExplainer`) para desglosar el peso exacto de cada variable sobre la predicción final.
   * **Traducción a Negocio:** Un motor de **Generación de Lenguaje Natural (NLG)** determinista que redacta la conclusión del análisis financiero.
5. **Capa de Persistencia (Cloud):** Conexión segura mediante *Service Accounts* a **Google BigQuery** (`pandas-gbq`), registrando cada evaluación productiva para alimentar el ecosistema de *Business Intelligence* en Looker Studio.

## 🚀 Cómo ejecutarlo localmente
1. Clona el repositorio:
```bash
git clone https://github.com/jlandaa/SmartCredit-ML
```
2. Instala las dependencias:
```bash
pip install -r requirements.txt
```
3. Ejecuta la aplicación de Streamlit:
```bash
streamlit run app.py
```

*(Nota: Para ejecutar la aplicación localmente con persistencia de datos, deberás configurar tus credenciales de Google BigQuery en el archivo .streamlit/secrets.toml. De lo contrario, puedes explorar la interfaz comentando la línea de carga de credenciales y activando el Modo Simulación).*

## 👨‍💻 Autor
## Juan Manuel Landa - Ingeniero en Computación
