import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Predicción Cardíaca con Red Neuronal",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Sistema de Predicción de Enfermedad Cardíaca (CNN / RNN)")
st.write("Despliegue con Red Neuronal Profunda, Función de Activación ReLU + Sigmoid y Escalado Dinámico.")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("modelo_nn_heart.keras")
    scaler = joblib.load("scaler_heart.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    with st.expander("ℹ️ Arquitectura y Funciones de Activación del Modelo", expanded=False):
        st.write(f"**Arquitectura Activa:** `{model.name}`")
        st.write("**Activación Capas Ocultas:** `ReLU` (o `Tanh` en RNN) — *Extrae patrones no lineales*")
        st.write("**Activación Capa de Salida:** `Sigmoid` — *Devuelve probabilidad de riesgo entre 0% y 100%*")
        st.write("**Estado:** Modelo y Scaler operativos ✅")
except Exception as e:
    st.error(f"Error al cargar archivos del modelo: {e}")
    st.stop()

tab_pred, tab_graficas = st.tabs([
    "📋 Realizar Predicción",
    "📊 Métricas y Función de Activación"
])

with tab_pred:
    st.subheader("Ingrese los datos clínicos del paciente:")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Edad (age)", min_value=1, max_value=120, value=54)
        sex = st.selectbox("Sexo (sex)", options=[0, 1], format_func=lambda x: "Femenino (0)" if x == 0 else "Masculino (1)")
        cp = st.selectbox("Tipo de dolor en el pecho (cp)", options=[1, 2, 3, 4], format_func=lambda x: {
            1: "1: Angina Típica", 2: "2: Angina Atípica", 3: "3: Dolor No Anginoso", 4: "4: Asintomático"
        }[x])
        trestbps = st.number_input("Presión arterial en reposo mm Hg (trestbps)", value=130)
        chol = st.number_input("Colesterol sérico mg/dl (chol)", value=240)
        fbs = st.selectbox("Azúcar en sangre en ayunas > 120 mg/dl (fbs)", options=[0, 1])
        restecg = st.selectbox("Resultados electrocardiógrafo en reposo (restecg)", options=[0, 1, 2])
    with col2:
        thalach = st.number_input("Frecuencia cardíaca máxima alcanzada (thalach)", value=150)
        exang = st.selectbox("Angina inducida por ejercicio (exang)", options=[0, 1])
        oldpeak = st.number_input("Depresión del ST (oldpeak)", value=1.0, step=0.1)
        slope = st.selectbox("Pendiente del segmento ST (slope)", options=[1, 2, 3])
        ca = st.selectbox("Número de vasos principales coloreados (ca)", options=[0, 1, 2, 3])
        thal = st.selectbox("Talasemia (thal)", options=[3.0, 6.0, 7.0], format_func=lambda x: {
            3.0: "Normal (3)", 6.0: "Defecto Fijo (6)", 7.0: "Defecto Reversible (7)"
        }[x])

    input_df = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])

    if st.button("Realizar Predicción con Red Neuronal"):
        # 1. Escalar con el StandardScaler entrenado
        scaled_array = scaler.transform(input_df)
        # 2. Redimensionar a 3D (1 muestra, 13 variables, 1 canal)
        input_3d = scaled_array.reshape((1, scaled_array.shape[1], 1))
        
        # 3. Predicción (salida de la función Sigmoid)
        probabilidad = float(model.predict(input_3d, verbose=0)[0][0])
        prediccion = 1 if probabilidad >= 0.5 else 0

        st.metric(label="Probabilidad estimada (Salida Sigmoid)", value=f"{probabilidad * 100:.2f}%")
        if prediccion == 1:
            st.error("⚠️ La Red Neuronal indica un **ALTO RIESGO** de enfermedad cardíaca.")
        else:
            st.success("✅ La Red Neuronal indica un **BAJO RIESGO** de enfermedad cardíaca.")

with tab_graficas:
    st.subheader("📊 Visualización de Activación y Desempeño")
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("### 1. Función de Activación (ReLU y Sigmoid)")
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        z = np.linspace(-5, 5, 200)
        ax1.plot(z, np.maximum(0, z), label="ReLU (Oculta)", linewidth=2)
        ax1.plot(z, 1 / (1 + np.exp(-z)), label="Sigmoid (Salida)", linewidth=2, linestyle="--")
        ax1.axhline(0.5, color="red", linestyle=":", label="Umbral decisión (0.5)")
        ax1.set_title("Funciones de Activación")
        ax1.set_xlabel("Entrada (z)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with col_g2:
        st.markdown("### 2. Distribución de Pacientes en Dataset")
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x=["Sano (0)", "Riesgo (1)"], y=[164, 139], palette="Set2", ax=ax2)
        ax2.set_title("Clases del Dataset Heart Disease")
        ax2.set_ylabel("Cantidad")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(fig2)
