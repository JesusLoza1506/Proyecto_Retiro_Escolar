import pandas as pd
import streamlit as st

# Importamos los renderizadores modulares independientes
from calidad import render_calidad
from correccion import render_correccion
from target import render_target
from categorical import render_categorical
from numericas import render_numericas
from classification import render_classification  # <-- NUEVA IMPORTACIÓN

# 1. Configuración de la interfaz
st.set_page_config(
    page_title="Predicción Retiro Escolar 2026", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Menú Abierto con Radio Buttons en el Panel Izquierdo
with st.sidebar:
    st.markdown("## 🎓 MINEDU")
    st.markdown("### ⚙️ Pipeline del Modelo")
    
    fase_seleccionada = st.radio(
        "Navegar por las etapas de la auditoría:",
        [
            "📋 Paso 1 - Diagnóstico de Calidad", 
            "🛠️ Paso 2 - Reglas de Corrección",
            "🎯 Paso 3 - Distribución del Target",
            "📊 Paso 4 - Variables Categóricas",
            "📈 Paso 5 - Variables Numéricas",
            "🤖 Paso 6 - Clasificación Binaria"  # <-- NUEVA PESTAÑA DE MACHINE LEARNING
        ],
        index=0
    )
    
    st.divider()
    st.caption("Fase activa actualmente cargada en el dashboard.")

# 3. Cache y Carga Optimizada de la Data
@st.cache_data
def cargar_datos_master():
    return pd.read_csv('Results_2026-05-31-2354.csv')

try:
    df = cargar_datos_master()
    
    # Orquestación según la selección abierta
    if "Paso 1" in fase_seleccionada:
        render_calidad(df)
    elif "Paso 2" in fase_seleccionada:
        render_correccion(df)
    elif "Paso 3" in fase_seleccionada:
        render_target(df)
    elif "Paso 4" in fase_seleccionada:
        render_categorical(df)
    elif "Paso 5" in fase_seleccionada:
        render_numericas(df)
    elif "Paso 6" in fase_seleccionada:
        render_classification(df)  # <-- ENRUTAMIENTO DINÁMICO DE CLASIFICACIÓN

except FileNotFoundError:
    st.error("❌ ¡Error Crítico! No se encontró el archivo 'Results_2026-05-31-2354.csv'.")