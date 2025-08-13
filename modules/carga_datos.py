# modules/carga_datos.py
import streamlit as st
import pandas as pd
from utils.supabase_conn import conectar_supabase

def cargar_datos(tabla: str) -> pd.DataFrame:
    supabase = conectar_supabase()
    response = supabase.table(tabla).select("*").execute()
    data = response.data
    return pd.DataFrame(data)

def mostrar_carga_datos():
    st.title("📥 Módulo 2: Carga de Datos")
    st.write("Conexión directa a Supabase")

    tabla = st.text_input("Nombre de la tabla en Supabase", value="vista_estudiantes_detalle", key="tabla_input")

    if st.button("Cargar datos"):
        try:
            df = cargar_datos(tabla)
            st.success(f"{len(df)} registros cargados desde '{tabla}'")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
