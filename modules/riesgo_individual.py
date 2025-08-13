# modules/riesgo_individual.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from typing import List, Tuple, Optional

# ---------------------------
# Utilidades internas
# ---------------------------

def _get_expected_columns_from_ct(ct) -> Tuple[List[str], List[str]]:
    """
    Devuelve (numericas_originales, categoricas_originales) tal como fueron
    registradas en el ColumnTransformer de entrenamiento.
    """
    cols_num, cols_cat = [], []
    for name, trans, cols in ct.transformers_:
        if name == "num":
            cols_num = list(cols)
        elif name == "cat":
            cols_cat = list(cols)
    return cols_num, cols_cat

def _load_model_safe(path: str):
    try:
        model = joblib.load(path)
        return model, None
    except FileNotFoundError:
        return None, f"❌ No se encontró el archivo de modelo: {path}"
    except Exception as e:
        return None, f"❌ Error al cargar el modelo: {e}"

def _build_single_input_from_df(df: pd.DataFrame,
                                cols_num: List[str],
                                cols_cat: List[str],
                                row: pd.Series) -> pd.DataFrame:
    """
    Construye un DataFrame de una fila con las columnas esperadas por el pipeline.
    """
    datos = {}
    for c in cols_num + cols_cat:
        if c in row:
            datos[c] = [row[c]]
        else:
            # Si faltara alguna columna, intentamos con NaN (el OHE ignora/scale maneja)
            datos[c] = [np.nan]
    return pd.DataFrame(datos)

def _build_manual_input_ui(df: pd.DataFrame,
                           cols_num: List[str],
                           cols_cat: List[str]) -> pd.DataFrame:
    """
    Construye un registro de 1 fila pidiendo valores al usuario.
    Para categóricas propone valores del DF si existen; si no, text_input.
    """
    st.markdown("#### ✍️ Ingreso manual de atributos")
    valores = {}
    # numéricas
    for col in cols_num:
        # sugiere rango según datos si están presentes
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            vmin = float(np.nanpercentile(df[col].values, 1)) if df[col].notna().any() else 0.0
            vmax = float(np.nanpercentile(df[col].values, 99)) if df[col].notna().any() else 100.0
            default = float(np.nanmedian(df[col].values)) if df[col].notna().any() else 0.0
            valores[col] = st.number_input(f"{col}", value=default, min_value=vmin, max_value=vmax, step=(vmax-vmin)/100 if vmax>vmin else 1.0)
        else:
            valores[col] = st.number_input(f"{col}", value=0.0)
    # categóricas
    for col in cols_cat:
        if col in df.columns:
            uniques = [u for u in df[col].dropna().unique().tolist() if u != ""]
            if 0 < len(uniques) <= 50:
                valores[col] = st.selectbox(f"{col}", options=uniques)
            else:
                valores[col] = st.text_input(f"{col}", value=str(uniques[0]) if uniques else "")
        else:
            valores[col] = st.text_input(f"{col}", value="")
    return pd.DataFrame([valores])

# ---------------------------
# Módulo principal
# ---------------------------

def mostrar_riesgo_individual(df: pd.DataFrame):
    """
    Vista Streamlit: predicción de riesgo individual usando el modelo .pkl entrenado.
    Requiere que el .pkl sea un Pipeline sklearn con:
        - step 'preprocesamiento' = ColumnTransformer(num, cat)
        - step 'modelo' con predict_proba
    """
    st.title("🧩 Módulo 5: Riesgo Individual")
    st.caption("Predice la probabilidad de deserción para un estudiante en particular.")

    # Ruta del modelo
    col_a, col_b = st.columns([2,1])
    with col_a:
        model_path = st.text_input("Ruta del modelo (.pkl)", value="modelo_reten_ia.pkl")
    with col_b:
        recargar = st.button("🔄 Recargar modelo", key="btn_reload_model")

    modelo, err = _load_model_safe(model_path)
    if err:
        st.error(err)
        st.stop()

    # Chequeo del pipeline esperado
    try:
        pre = modelo.named_steps["preprocesamiento"]
        cls = modelo.named_steps["modelo"]
    except Exception:
        st.error("El archivo .pkl debe ser un Pipeline con steps: 'preprocesamiento' y 'modelo'.")
        st.stop()

    cols_num, cols_cat = _get_expected_columns_from_ct(pre)
    st.markdown("###### Columnas esperadas por el modelo")
    st.write({"numéricas": cols_num, "categóricas": cols_cat})

    modo = st.radio("Modo de ingreso de datos:", ["Seleccionar del dataset", "Ingreso manual"], horizontal=True)

    # Umbral de decisión
    th = st.slider("🎚️ Umbral de decisión (prob. ≥ umbral ⇒ 'Deserta')",
                   min_value=0.05, max_value=0.95, value=0.50, step=0.01)

    # --- Selección desde dataset ---
    if modo == "Seleccionar del dataset":
        st.markdown("#### 🔎 Selecciona un estudiante del dataset")
        id_col_candidates = [c for c in ["codigo_estudiante", "id", "codigo", "dni"] if c in df.columns]
        id_col = id_col_candidates[0] if id_col_candidates else None

        if id_col:
            opciones = df[id_col].astype(str).tolist()
            seleccionado = st.selectbox(f"Identificador ({id_col})", options=opciones)
            row = df[df[id_col].astype(str) == seleccionado].iloc[0]
        else:
            st.info("No se encontró columna identificadora típica (p.ej., 'codigo_estudiante'). Se usará el índice.")
            idx = st.number_input("Índice de fila", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
            row = df.iloc[int(idx)]

        X1 = _build_single_input_from_df(df, cols_num, cols_cat, row)

    # --- Ingreso manual ---
    else:
        X1 = _build_manual_input_ui(df, cols_num, cols_cat)

    st.markdown("---")
    if st.button("📌 Calcular Riesgo", key="btn_calcular_riesgo"):
        try:
            proba = float(modelo.predict_proba(X1)[0, 1])
            pred = int(proba >= th)

            st.subheader("🧠 Resultado")
            st.metric("Probabilidad de Deserción", f"{proba:.3f}")
            st.metric("Clasificación (según umbral)", "Deserta (1)" if pred == 1 else "Retenido (0)")

            # Barra visual
            st.progress(min(max(proba, 0.0), 1.0))

            with st.expander("Ver entrada utilizada (X)"):
                st.dataframe(X1, use_container_width=True)

            # Log opcional
            if st.toggle("Guardar resultado en CSV (logs/predicciones.csv)", value=False, key="toggle_log"):
                import os
                os.makedirs("logs", exist_ok=True)
                out = X1.copy()
                out["prob_desercion"] = proba
                out["pred_desercion"] = pred
                out["umbral"] = th
                out.to_csv("logs/predicciones.csv", mode="a", header=not os.path.exists("logs/predicciones.csv"), index=False)
                st.success("✅ Resultado guardado en logs/predicciones.csv")

        except Exception as e:
            st.error(f"No se pudo generar la predicción: {e}")
