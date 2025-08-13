# app.py
import streamlit as st
from modules.carga_datos import cargar_datos, mostrar_carga_datos
from modules.estadisticas import mostrar_estadisticas
from modules.eda import mostrar_eda
from modules.modelado import mostrar_modelado
from modules.dashboard import mostrar_dashboard
from modules.riesgo_individual import mostrar_riesgo_individual

# Sidebar con opciones de módulos
st.sidebar.title("🔍 Menú Principal")
opcion = st.sidebar.radio("Selecciona un módulo", (
    "Carga de Datos",
    "Estadísticas Descriptivas",
    "Exploración de Datos (EDA)",
    "Entrenamiento del Modelo",
    "Riesgo Individual",
    "Dashboard",
    "Configuración"
))

# Inicializamos df como None por defecto
df = None

# Si no es la opción de carga, cargamos automáticamente los datos
if opcion != "Carga de Datos":
    df = cargar_datos("vista_estudiantes_detalle")

# Módulo: Carga de Datos
if opcion == "Carga de Datos":
    mostrar_carga_datos()

# Módulo: Estadísticas Descriptivas
elif opcion == "Estadísticas Descriptivas":
    if df is not None:
        mostrar_estadisticas(df)
    else:
        st.warning("Primero carga los datos.")

# Módulo: Exploración de Datos (EDA)
elif opcion == "Exploración de Datos (EDA)":
    if df is not None:
        mostrar_eda(df)
    else:
        st.warning("Primero carga los datos.")

elif opcion == "Entrenamiento del Modelo":
    modelo = mostrar_modelado(df)

elif opcion == "Riesgo Individual":
    mostrar_riesgo_individual(df)
    

elif opcion == "Dashboard":
    if df is not None:
        mostrar_dashboard(df)
    else:
        st.warning("Primero carga los datos.")



