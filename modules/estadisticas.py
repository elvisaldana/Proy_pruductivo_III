# modules/estadisticas.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.supabase_conn import conectar_supabase

def mostrar_estadisticas(df):
    st.title("Estadísticas Descriptivas")

    supabase = conectar_supabase()
    response = supabase.table("vista_estudiantes_detalle").select("*").execute()
    data = response.data
    df = pd.DataFrame(data)

    # Gráfico 1: Deserción por ciclo
    st.subheader("Cantidad de deserciones por ciclo académico")
    desertores = df[df["deserto"] == True]
    resumen = (
        desertores.groupby("ciclo_desercion")
        .size()
        .reset_index(name="cantidad")
        .sort_values("ciclo_desercion")
    )
    fig1 = px.bar(
        resumen,
        x="ciclo_desercion",
        y="cantidad",
        labels={"ciclo_desercion": "Ciclo de Deserción", "cantidad": "Cantidad de Estudiantes"},
        title="Cantidad de deserciones por ciclo académico",
        text="cantidad"
    )
    fig1.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        title_font_size=20
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Gráfico 2: Porcentaje desertores vs activos
    mostrar_porcentaje_desercion(df)

        # Gráfico 3: Comparativa por atributo
    mostrar_comparativa_desercion(df)

    mostrar_desercion_por_nivel_socioeconomico(df)




def mostrar_porcentaje_desercion(df):
    st.subheader("Porcentaje de estudiantes desertores vs activos")

    conteo = df["deserto"].value_counts().reset_index()
    conteo.columns = ["deserto", "cantidad"]

    # Etiquetas legibles
    conteo["estado"] = conteo["deserto"].map({True: "Desertores", False: "Activos"})

    fig = px.pie(
        conteo,
        values="cantidad",
        names="estado",
        title="Distribución de estudiantes según estado",
        color_discrete_sequence=["#EF4444", "#10B981"]
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        title_font_size=20
    )

    st.plotly_chart(fig, use_container_width=True)



def mostrar_comparativa_desercion(df):
    st.subheader("Comparativa de deserciones por atributo")

    atributo = st.selectbox("Selecciona un atributo para comparar", ["sexo", "modalidad", "turno", "carrera"])

    if atributo in df.columns:
        df_grouped = df.groupby(atributo).size().reset_index(name="cantidad")

        st.write("Vista previa de agrupación:", df_grouped)

        fig = px.bar(
            df_grouped,
            x=atributo,
            y="cantidad",
            color=atributo,
            title=f"Comparativa de deserciones por {atributo.capitalize()}",
            labels={atributo: atributo.capitalize(), "cantidad": "Cantidad de Estudiantes"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"El atributo '{atributo}' no se encuentra en el DataFrame.")


def mostrar_desercion_por_nivel_socioeconomico(df):
    st.markdown("### Deserción según Nivel Socioeconómico")

    df_grouped = df.groupby("nivel_socioeconomico").size().reset_index(name="cantidad")

    fig = px.bar(
        df_grouped,
        x="nivel_socioeconomico",
        y="cantidad",
        color="nivel_socioeconomico",
        text="cantidad",
        title="Cantidad de deserciones por Nivel Socioeconómico",
        labels={"nivel_socioeconomico": "Nivel Socioeconómico", "cantidad": "Cantidad de Estudiantes"}
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)
