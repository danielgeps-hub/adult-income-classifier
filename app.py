"""
Adult Income Classifier - Streamlit app
Publicacion del modelo de clasificacion desarrollado en la Solemne 1
del Taller de Aplicaciones (UCI Adult / Census Income).

Modelo: Gradient Boosting tuneado con GridSearchCV + balanceo de clases.
F1 (>50K) = 0.7118  |  ROC-AUC = 0.9277  |  n_test = 16,281
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURACION DE PAGINA
# ============================================================
st.set_page_config(
    page_title="Adult Income Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CARGA DEL BUNDLE (cacheada para no recargar en cada interaccion)
# ============================================================
@st.cache_resource
def cargar_bundle(ruta: str = "modelo_v2_bundle.pkl") -> dict:
    """Carga el modelo y todos sus metadatos desde el .pkl."""
    return joblib.load(ruta)


bundle = cargar_bundle()
clf            = bundle["modelo"]
feat_cols      = bundle["feature_columns"]
categorias     = bundle["categorias"]
rangos         = bundle["rangos_numericos"]
metricas       = bundle["metricas"]
top_features   = bundle["top_features"]
best_params    = bundle["best_params"]
cm             = np.array(bundle["matriz_confusion"])
roc_data       = bundle["roc_data"]
tabla_modelos  = pd.DataFrame.from_dict(bundle["tabla_modelos"], orient="index")


# ============================================================
# PREPROCESAMIENTO (replicado identico al notebook)
# ============================================================
def preprocesar_adult(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el mismo preprocesamiento que se uso en entrenamiento."""
    df = df.copy()
    df["income"] = df["income"].astype(str).str.rstrip(".")
    for col in ["workclass", "occupation", "native-country"]:
        df[col] = df[col].fillna("Unknown")
    df = df.drop(columns=["fnlwgt", "education"], errors="ignore")
    df["workclass"] = df["workclass"].replace(
        ["Without-pay", "Never-worked"], "Unknown"
    )
    marital_map = {
        "Married-civ-spouse":   "Married",
        "Married-AF-spouse":    "Married",
        "Married-spouse-absent": "Separated",
        "Separated":            "Separated",
        "Divorced":             "Divorced",
        "Widowed":              "Widowed",
        "Never-married":        "Never-married",
    }
    df["marital-status"] = df["marital-status"].map(marital_map)
    df["native-country"] = df["native-country"].apply(
        lambda x: "United-States" if x == "United-States"
        else ("Unknown" if x == "Unknown" else "Other")
    )
    df["capital-gain"] = df["capital-gain"].replace(99999, 41310)
    df["has_capital_gain"] = (df["capital-gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital-loss"] > 0).astype(int)
    return df


def predecir(caso: dict) -> float:
    """Pipeline completo: dict crudo -> probabilidad de >50K."""
    df = pd.DataFrame([caso])
    df_clean = preprocesar_adult(df)
    X = df_clean.drop(columns=["income"])
    X_enc = pd.get_dummies(X).astype(int)
    X_aligned = X_enc.reindex(columns=feat_cols, fill_value=0)
    return float(clf.predict_proba(X_aligned)[0, 1])


# ============================================================
# SIDEBAR — informacion del proyecto
# ============================================================
with st.sidebar:
    st.title("Adult Income Classifier")
    st.markdown(
        "**Publicacion del modelo de clasificacion** desarrollado en la "
        "Solemne 1 del Taller de Aplicaciones."
    )
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown(
        "**Adult / Census Income** (UCI Machine Learning Repository). "
        "Censo de EE.UU. 1994, 48,842 personas. El objetivo es predecir si "
        "el ingreso anual supera los **USD 50,000**."
    )
    st.markdown("### Modelo final")
    st.markdown(
        "**Gradient Boosting** tuneado con `GridSearchCV` (8 combinaciones x "
        "5 folds) y balanceo de clases via `sample_weight`."
    )
    st.markdown("### Hiperparametros ganadores")
    st.code(
        f"n_estimators  = {best_params['n_estimators']}\n"
        f"max_depth     = {best_params['max_depth']}\n"
        f"learning_rate = {best_params['learning_rate']}",
        language="text",
    )
    st.markdown("---")
    st.caption(
        "Trabajo academico — Universidad San Sebastian.  \n"
        "Taller de Aplicaciones · 2026."
    )


# ============================================================
# HEADER
# ============================================================
st.title("Clasificador de Ingresos — Adult Census Income")
st.markdown(
    "Esta aplicacion publica el clasificador desarrollado en la Solemne 1. "
    "Permite (1) revisar el rendimiento del modelo sobre el set de test y "
    "(2) probar el modelo ingresando los datos de una persona."
)

tab_metricas, tab_prueba, tab_tecnica = st.tabs([
    "Rendimiento del modelo",
    "Probar el clasificador",
    "Ficha tecnica",
])


# ============================================================
# TAB 1 — RENDIMIENTO DEL MODELO
# ============================================================
with tab_metricas:
    st.subheader("Metricas sobre el set de test")
    st.caption(
        f"Evaluado sobre {metricas['n_test']:,} muestras del archivo "
        "`adult.test`, no vistas durante el entrenamiento."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",       f"{metricas['accuracy']:.4f}")
    c2.metric("Precision >50K", f"{metricas['precision']:.4f}")
    c3.metric("Recall >50K",    f"{metricas['recall']:.4f}")
    c4.metric("F1 >50K",        f"{metricas['f1']:.4f}")
    c5.metric("ROC-AUC",        f"{metricas['roc_auc']:.4f}")

    st.markdown(
        "El balanceo de clases sube el **recall** sobre `>50K` "
        f"(detecta {metricas['recall']*100:.1f}% de los casos reales positivos) "
        "a costa de algo de precision. F1 y ROC-AUC son las metricas relevantes "
        "porque el dataset esta desbalanceado (76% `<=50K` vs 24% `>50K`)."
    )

    st.markdown("---")

    col_cm, col_roc = st.columns(2)

    # --- Matriz de confusion ---
    with col_cm:
        st.markdown("#### Matriz de confusion")
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Pred <=50K", "Pred >50K"],
            y=["Real <=50K", "Real >50K"],
            text=cm,
            texttemplate="%{text:,}",
            textfont={"size": 18},
            colorscale="Blues",
            showscale=False,
        ))
        fig_cm.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        vn, fp, fn, vp = cm.ravel()
        st.caption(
            f"VN = {vn:,} · FP = {fp:,} · FN = {fn:,} · VP = {vp:,}"
        )

    # --- Curva ROC ---
    with col_roc:
        st.markdown("#### Curva ROC")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=roc_data["fpr"], y=roc_data["tpr"],
            mode="lines",
            name=f"modelo_v2 (AUC = {roc_data['auc']:.4f})",
            line=dict(color="#B85042", width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Aleatorio",
            line=dict(color="grey", width=1, dash="dash"),
        ))
        fig_roc.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.5, y=0.05, xanchor="center", bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # --- Comparacion entre modelos ---
    st.markdown("#### Comparacion con otros modelos evaluados")
    st.caption(
        "Las metricas en negativo del modelo final (Accuracy) son el costo "
        "esperado del balanceo de clases. F1 y ROC-AUC mejoraron."
    )

    columnas_mostrar = [
        "Accuracy", "Balanced Acc.", "Precision (>50K)",
        "Recall (>50K)", "F1 (>50K)", "ROC-AUC"
    ]
    tabla_show = tabla_modelos[columnas_mostrar].round(4)
    st.dataframe(
        tabla_show.style
            .highlight_max(axis=0, color="#d4edda")
            .format("{:.4f}", na_rep="—"),
        use_container_width=True,
    )

    # --- Importancia de variables ---
    st.markdown("#### Variables mas importantes (Gini gain)")
    df_imp = (
        pd.Series(top_features)
        .sort_values(ascending=True)
        .reset_index()
    )
    df_imp.columns = ["Variable", "Importancia"]

    fig_imp = px.bar(
        df_imp,
        x="Importancia",
        y="Variable",
        orientation="h",
        text=df_imp["Importancia"].round(3),
    )
    fig_imp.update_traces(marker_color="#5B8DB8", textposition="outside")
    fig_imp.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Importancia (Gini gain)",
        yaxis_title="",
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# ============================================================
# TAB 2 — PROBAR EL CLASIFICADOR
# ============================================================
with tab_prueba:
    st.subheader("Ingresa los datos de una persona")
    st.caption(
        "Todos los campos parten con valores de referencia (medianas/modas del set "
        "de entrenamiento). Modificalos para evaluar distintos perfiles."
    )

    with st.form("formulario_prediccion"):
        # --- Variables numericas ---
        st.markdown("##### Variables numericas")
        col_n1, col_n2, col_n3 = st.columns(3)
        with col_n1:
            age = st.slider(
                "Edad",
                int(rangos["age"][0]), int(rangos["age"][1]),
                value=38,
            )
            education_num = st.slider(
                "Anios de educacion (education-num)",
                int(rangos["education-num"][0]),
                int(rangos["education-num"][1]),
                value=10,
                help="Equivalencias: 9=HS-grad, 10=Some-college, 13=Bachelors, "
                     "14=Masters, 16=Doctorate.",
            )
        with col_n2:
            hours_per_week = st.slider(
                "Horas trabajadas por semana",
                int(rangos["hours-per-week"][0]),
                int(rangos["hours-per-week"][1]),
                value=40,
            )
            capital_gain = st.number_input(
                "Capital gain (USD anual)",
                min_value=int(rangos["capital-gain"][0]),
                max_value=int(rangos["capital-gain"][1]),
                value=0,
                step=500,
            )
        with col_n3:
            capital_loss = st.number_input(
                "Capital loss (USD anual)",
                min_value=int(rangos["capital-loss"][0]),
                max_value=int(rangos["capital-loss"][1]),
                value=0,
                step=100,
            )

        st.markdown("##### Variables categoricas")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            workclass = st.selectbox(
                "Tipo de empleo (workclass)",
                categorias["workclass"],
                index=categorias["workclass"].index("Private")
                if "Private" in categorias["workclass"] else 0,
            )
            occupation = st.selectbox(
                "Ocupacion",
                categorias["occupation"],
                index=categorias["occupation"].index("Prof-specialty")
                if "Prof-specialty" in categorias["occupation"] else 0,
            )
            marital_status = st.selectbox(
                "Estado civil (marital-status)",
                categorias["marital-status"],
                index=categorias["marital-status"].index("Never-married")
                if "Never-married" in categorias["marital-status"] else 0,
            )
            relationship = st.selectbox(
                "Relacion familiar (relationship)",
                categorias["relationship"],
                index=categorias["relationship"].index("Not-in-family")
                if "Not-in-family" in categorias["relationship"] else 0,
            )
        with col_c2:
            sex = st.selectbox(
                "Sexo",
                categorias["sex"],
                index=categorias["sex"].index("Male")
                if "Male" in categorias["sex"] else 0,
            )
            race = st.selectbox(
                "Raza",
                categorias["race"],
                index=categorias["race"].index("White")
                if "White" in categorias["race"] else 0,
            )
            native_country = st.selectbox(
                "Pais de origen",
                categorias["native-country"],
                index=categorias["native-country"].index("United-States")
                if "United-States" in categorias["native-country"] else 0,
            )

        submitted = st.form_submit_button(
            "Predecir ingreso", type="primary", use_container_width=True
        )

    if submitted:
        # Construir el caso crudo (formato identico al CSV original)
        caso = {
            "age":            age,
            "workclass":      workclass,
            "fnlwgt":         0,        # se descarta en preprocesar_adult
            "education":      "HS-grad",  # se descarta (redundante con education-num)
            "education-num":  education_num,
            "marital-status": marital_status,
            "occupation":     occupation,
            "relationship":   relationship,
            "race":           race,
            "sex":            sex,
            "capital-gain":   capital_gain,
            "capital-loss":   capital_loss,
            "hours-per-week": hours_per_week,
            "native-country": native_country,
            "income":         "<=50K",  # dummy, no se usa para predecir
        }

        proba = predecir(caso)
        clase = ">50K" if proba >= 0.5 else "<=50K"

        st.markdown("---")
        st.markdown("### Resultado")

        col_r1, col_r2 = st.columns([1, 2])

        with col_r1:
            color = "#2E8B57" if clase == ">50K" else "#8B6F47"
            st.markdown(
                f"<div style='text-align:center; padding:24px; "
                f"background-color:{color}; color:white; border-radius:12px;'>"
                f"<div style='font-size:14px; opacity:0.85;'>Prediccion</div>"
                f"<div style='font-size:34px; font-weight:700; margin-top:4px;'>"
                f"{clase}</div>"
                f"<div style='font-size:13px; opacity:0.85; margin-top:8px;'>"
                f"Probabilidad de >50K: {proba*100:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_r2:
            # Gauge / barra horizontal con la probabilidad
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": " %", "font": {"size": 36}},
                title={"text": "Probabilidad estimada de ingreso > USD 50,000"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 50],  "color": "#f5f5f5"},
                        {"range": [50, 100], "color": "#e8f4ea"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 2},
                        "thickness": 0.85,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=220, margin=dict(l=20, r=20, t=40, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Resumen de los inputs (para chequear que esten bien)
        with st.expander("Ver detalle de los datos ingresados"):
            df_caso = pd.DataFrame.from_dict(
                {k: [v] for k, v in caso.items() if k != "income"},
                orient="columns",
            ).T
            df_caso.columns = ["Valor"]
            st.dataframe(df_caso, use_container_width=True)

        st.caption(
            "Umbral de decision: 0.5. Una probabilidad >= 50% se clasifica "
            "como `>50K`. Este umbral puede ajustarse si se prioriza recall o "
            "precision segun el caso de uso."
        )


# ============================================================
# TAB 3 — FICHA TECNICA
# ============================================================
with tab_tecnica:
    st.subheader("Ficha tecnica del modelo")

    st.markdown("#### Pipeline de preprocesamiento")
    st.markdown(
        "- Imputacion de `NaN` con `'Unknown'` en `workclass`, `occupation` "
        "y `native-country`.\n"
        "- Eliminacion de `fnlwgt` (peso del censo, no es atributo del individuo) "
        "y `education` (redundante con `education-num`).\n"
        "- Agrupacion de categorias raras: `workclass={Without-pay, Never-worked} -> Unknown`; "
        "`marital-status` simplificado a 5 categorias; `native-country` reducido a "
        "`{United-States, Other, Unknown}`.\n"
        "- Winsorizacion de `capital-gain`: el valor centinela 99,999 se reemplaza por 41,310 "
        "(maximo real observado).\n"
        "- Variables binarias derivadas: `has_capital_gain`, `has_capital_loss`.\n"
        "- One-hot encoding sobre las 7 categoricas (drop_first=True), llegando a "
        f"**{len(feat_cols)} features** finales."
    )

    st.markdown("#### Algoritmo")
    st.markdown(
        "**`GradientBoostingClassifier`** de scikit-learn, seleccionado por tener "
        "el mejor F1 y ROC-AUC entre 5 candidatos (Logistic Regression, Decision Tree, "
        "KNN, Random Forest, Gradient Boosting)."
    )

    st.markdown("#### Tuning")
    st.markdown(
        "- `GridSearchCV` sobre `n_estimators` x `max_depth` x `learning_rate` "
        "(8 combinaciones).\n"
        "- `StratifiedKFold(5)` que preserva la proporcion de clases.\n"
        "- Scoring: **F1** (por desbalance de clases; accuracy serria enganioso).\n"
        "- Balanceo de clases via `sample_weight = compute_sample_weight('balanced', y_train)`."
    )

    st.markdown("#### Hiperparametros finales")
    df_params = pd.DataFrame.from_dict(best_params, orient="index", columns=["Valor"])
    st.dataframe(df_params, use_container_width=True)

    st.markdown("#### Mejoras futuras posibles")
    st.markdown(
        "- Explorar XGBoost / LightGBM (suelen superar a GB de sklearn con menor tiempo).\n"
        "- Ingenieria de caracteristicas adicional (interacciones, binning de `age`).\n"
        "- Ajustar el umbral de decision segun la metrica de negocio prioritaria.\n"
        "- Calibracion de probabilidades (`CalibratedClassifierCV`)."
    )

    st.markdown("---")
    st.markdown("#### Referencia del dataset")
    st.markdown(
        "Becker, B. & Kohavi, R. (1996). **Adult**. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5XW20"
    )
