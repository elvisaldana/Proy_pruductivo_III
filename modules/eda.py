
import streamlit as st
import plotly.express as px

def mostrar_eda(df):
    """
    Muestra el menú de selección de análisis EDA.
    """
    st.header("Exploración de Datos (EDA)")
    st.subheader("Análisis exploratorio de variables")

    eda_opcion = st.radio("Selecciona el tipo de análisis:", 
                          ["Distribución de variables numéricas", 
                           "Boxplots por nivel socioeconómico",
                           "Relación entre sexo y deserción",
                           "Relación entre nivel socioeconómico y deserción"])

    if eda_opcion == "Distribución de variables numéricas":
        mostrar_histogramas(df)
    elif eda_opcion == "Boxplots por nivel socioeconómico":
        boxplot_ingreso_por_nivel(df)
    elif eda_opcion == "Relación entre sexo y deserción":
        mostrar_desercion_por_sexo(df)
    elif eda_opcion == "Relación entre nivel socioeconómico y deserción":
        mostrar_desercion_por_nivel_socioeconomico(df)

# Función para graficar histogramas de variables numéricas
def mostrar_histogramas(df):
    st.subheader("Distribución de Variables Numéricas")
    variables_numericas = ["edad", "numero_hijos", "horas_trabajo_semanal", 
                           "ingreso_mensual"]

    for var in variables_numericas:
        if var in df.columns:
            fig = px.histogram(df, x=var, nbins=20, title=f"Distribución de {var}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"La variable '{var}' no existe en los datos.")

# Función para graficar boxplot de ingreso mensual según nivel socioeconómico
def boxplot_ingreso_por_nivel(df):
    st.subheader("Boxplot: Ingreso mensual por nivel socioeconómico")
    fig = px.box(df, x="nivel_socioeconomico", y="ingreso_mensual", 
                 points="all", color="nivel_socioeconomico")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Función para mostrar deserción por sexo
def mostrar_desercion_por_sexo(df):
    st.markdown("### Deserción según Sexo")
    df_grouped = df.groupby("sexo").size().reset_index(name="cantidad")
    fig = px.bar(
        df_grouped,
        x="sexo",
        y="cantidad",
        color="sexo",
        text="cantidad",
        title="Cantidad de deserciones por Sexo",
        labels={"sexo": "Sexo", "cantidad": "Cantidad de Estudiantes"}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)

# Función para mostrar deserción por nivel socioeconómico
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

