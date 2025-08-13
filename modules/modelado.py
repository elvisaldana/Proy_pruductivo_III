# modules/modelado.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_curve, roc_auc_score
import plotly.express as px
import joblib
import matplotlib.pyplot as plt

# Opcional: XGBoost
try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

# -----------------------------------------------
# FUNCIÓN 1: Selección de Variables
# -----------------------------------------------
def seleccionar_variables(df):
    st.subheader("🎯 Selección de Variables")
    columnas_disponibles = df.columns.tolist()
    st.write("Columnas disponibles:", columnas_disponibles)

    variables_x = st.multiselect(
        "Selecciona las variables predictoras (X):",
        options=columnas_disponibles,
        default=["edad", "nivel_socioeconomico", "horas_trabajo_semanal", "ingreso_mensual", "estado_civil", "trabaja", "numero_hijos", "sexo"]
    )

    variable_y = "deserto"
    if variable_y not in df.columns:
        st.error("La variable objetivo 'deserto' no se encuentra en el DataFrame.")
        return None, None

    if not variables_x:
        st.warning("Selecciona al menos una variable independiente.")
        return None, None

    return df[variables_x], df[variable_y]

# -----------------------------------------------
# FUNCIÓN 2: Pipeline de Preprocesamiento
# -----------------------------------------------
def construir_pipeline(X):
    columnas_numericas = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    columnas_categoricas = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocesador = ColumnTransformer(transformers=[
        ('num', StandardScaler(), columnas_numericas),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), columnas_categoricas)
    ])

    return preprocesador, columnas_numericas, columnas_categoricas

# -----------------------------------------------
# FUNCIÓN 3: Entrenamiento
# -----------------------------------------------
def entrenar_modelo(X, y):
    st.subheader("📈 Entrenamiento del Modelo")

    modelos_disponibles = {
        "Regresión Logística": LogisticRegression(max_iter=1000),
        "Árbol de Decisión": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    if xgboost_available:
        modelos_disponibles["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    modelo_nombre = st.selectbox("Selecciona el modelo:", list(modelos_disponibles.keys()))
    split = st.slider("Porcentaje para entrenamiento", 60, 90, 80, 5)
    test_size = 1 - split / 100

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    preprocesador, cols_num, cols_cat = construir_pipeline(X)
    modelo_final = Pipeline([
        ("preprocesamiento", preprocesador),
        ("modelo", modelos_disponibles[modelo_nombre])
    ])

     
    if st.button("🚀 Entrenar Modelo"):
        modelo_final.fit(X_train, y_train)
        y_pred = modelo_final.predict(X_test)

        evaluar_modelo(y_test, y_pred)
        mostrar_curva_roc(modelo_final, X_test, y_test)
        guardar_modelo(modelo_final, "modelo_reten_ia.pkl")

        st.success(f"✅ Modelo '{modelo_nombre}' entrenado y guardado como 'modelo_reten_ia.pkl'.")
        return modelo_final


# -----------------------------------------------
# FUNCIÓN 4: Evaluación
# -----------------------------------------------
def evaluar_modelo(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    st.markdown("### 📊 Métricas de Evaluación")
    st.write(f"- **Accuracy**: {acc:.3f}")
    st.write(f"- **Precision**: {prec:.3f}")
    st.write(f"- **Recall**: {rec:.3f}")
    st.write(f"- **F1-Score**: {f1:.3f}")

    cm_df = pd.DataFrame(cm, index=["Retenido (0)", "Desertó (1)"],
                         columns=["Predicho 0", "Predicho 1"])
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="blues", title="Matriz de Confusión")
    st.plotly_chart(fig)

# -----------------------------------------------
# FUNCIÓN 5: Guardar/Cargar modelo
# -----------------------------------------------
def guardar_modelo(modelo, nombre_archivo):
    joblib.dump(modelo, nombre_archivo)

def cargar_modelo(nombre_archivo):
    try:
        modelo = joblib.load(nombre_archivo)
        return modelo
    except FileNotFoundError:
        st.error(f"Archivo '{nombre_archivo}' no encontrado.")
        return None

# -----------------------------------------------
# FUNCIÓN PRINCIPAL PARA STREAMLIT
# -----------------------------------------------
def mostrar_modelado(df):
    st.title("🧠 Módulo 4: Modelado Predictivo - RETEN_IA2")
    st.markdown("Entrena, evalúa y guarda modelos de predicción de deserción estudiantil.")

    X, y = seleccionar_variables(df)
    if X is not None and y is not None:
        modelo_entrenado = entrenar_modelo(X, y)
        return modelo_entrenado
    else:
        return None

# -----------------------------------------------
# función para mostrar curva ROC
# -----------------------------------------------
def mostrar_curva_roc(modelo, X_test, y_test):
    try:
        # Obtener probabilidades para clase positiva (1)
        y_proba = modelo.predict_proba(X_test)[:, 1]

        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        # Mostrar gráfico
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color="blue")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("Tasa de falsos positivos (FPR)")
        ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
        ax.set_title("Curva ROC")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"No se pudo calcular la curva ROC: {e}")
