"""
Adult Income Classifier - Streamlit app (V2 con mejoras)
Publicacion del modelo de clasificacion desarrollado en la Solemne 1
del Taller de Aplicaciones (UCI Adult / Census Income).

Modelo: Gradient Boosting tuneado con GridSearchCV + balanceo de clases.
Mejoras V2: umbral dinamico, contribucion por feature, analisis de equidad.
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURACION
# ============================================================
st.set_page_config(
    page_title="Adult Income Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def cargar_bundle(ruta: str = "modelo_v2_bundle.pkl") -> dict:
    return joblib.load(ruta)


bundle = cargar_bundle()
clf            = bundle["modelo"]
feat_cols      = bundle["feature_columns"]
categorias     = bundle["categorias"]
rangos         = bundle["rangos_numericos"]
top_features   = bundle["top_features"]
best_params    = bundle["best_params"]
roc_data       = bundle["roc_data"]
tabla_modelos  = pd.DataFrame.from_dict(bundle["tabla_modelos"], orient="index")
medias_clase   = bundle["medias_por_clase"]
curvas         = bundle["curvas_umbral"]
fairness       = bundle["fairness"]
N_TEST         = bundle["metricas"]["n_test"]

umbrales_arr   = np.array(curvas["umbral"])


# ============================================================
# PREPROCESAMIENTO E INFERENCIA
# ============================================================
def preprocesar_adult(df: pd.DataFrame) -> pd.DataFrame:
    """Replica identica del pipeline de entrenamiento."""
    df = df.copy()
    df["income"] = df["income"].astype(str).str.rstrip(".")
    for col in ["workclass", "occupation", "native-country"]:
        df[col] = df[col].fillna("Unknown")
    df = df.drop(columns=["fnlwgt", "education"], errors="ignore")
    df["workclass"] = df["workclass"].replace(
        ["Without-pay", "Never-worked"], "Unknown"
    )
    marital_map = {
        "Married-civ-spouse":    "Married",
        "Married-AF-spouse":     "Married",
        "Married-spouse-absent": "Separated",
        "Separated":             "Separated",
        "Divorced":              "Divorced",
        "Widowed":               "Widowed",
        "Never-married":         "Never-married",
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


def construir_X(caso: dict) -> pd.DataFrame:
    df_clean = preprocesar_adult(pd.DataFrame([caso]))
    X = df_clean.drop(columns=["income"])
    X_enc = pd.get_dummies(X).astype(int)
    return X_enc.reindex(columns=feat_cols, fill_value=0)


def predecir_proba(caso: dict):
    X = construir_X(caso)
    proba = float(clf.predict_proba(X)[0, 1])
    return proba, X


def calcular_contribuciones(X_row: pd.Series, top_n: int = 8) -> pd.DataFrame:
    """Para cada feature, estima cuanto empuja la prediccion hacia >50K o <=50K.

    push = (x_usuario - punto_medio_entre_clases) * direccion * importancia
    donde direccion = +1 si la feature tiene media mayor en >50K, -1 si la
    tiene mayor en <=50K. Una feature con valor cercano a la media de >50K
    y direccion positiva produce un push positivo (empuja hacia >50K).
    """
    media_pos = pd.Series(medias_clase[">50K"])
    media_neg = pd.Series(medias_clase["<=50K"])
    midpoint = (media_pos + media_neg) / 2
    direction = np.sign(media_pos - media_neg).replace(0, 1)

    importancias_all = pd.Series(clf.feature_importances_, index=feat_cols)

    contribs = []
    for feat in feat_cols:
        x_val = float(X_row[feat])
        mp = float(midpoint.get(feat, 0))
        dr = float(direction.get(feat, 1))
        imp = float(importancias_all[feat])
        push = (x_val - mp) * dr * imp
        contribs.append({
            "feature": feat,
            "valor_usuario": x_val,
            "media_<=50K": float(media_neg.get(feat, 0)),
            "media_>50K":  float(media_pos.get(feat, 0)),
            "push": push,
            "importancia_global": imp,
        })

    df = pd.DataFrame(contribs)
    df["abs_push"] = df["push"].abs()
    return df.sort_values("abs_push", ascending=False).head(top_n)


# ============================================================
# SIDEBAR (incluye control global de umbral)
# ============================================================
with st.sidebar:
    st.title("Adult Income Classifier")
    st.markdown("Publicacion del modelo de la Solemne 1 — Taller de Aplicaciones.")
    st.markdown("---")

    st.markdown("### Umbral de decision")
    umbral = st.slider(
        "Probabilidad minima para predecir >50K",
        min_value=0.05, max_value=0.95, value=0.50, step=0.01,
        help="Por defecto 0.5. Mover el slider muestra como cambian las metricas "
             "cuando se exige mas o menos probabilidad para clasificar como >50K. "
             "Las metricas y la matriz de confusion del tab 'Rendimiento' se "
             "recalculan en vivo.",
    )

    idx_u  = int(np.argmin(np.abs(umbrales_arr - umbral)))
    idx_05 = int(np.argmin(np.abs(umbrales_arr - 0.5)))

    delta_f1 = curvas["f1"][idx_u] - curvas["f1"][idx_05]
    st.metric(
        f"F1 (>50K) @ umbral {curvas['umbral'][idx_u]:.2f}",
        f"{curvas['f1'][idx_u]:.4f}",
        delta=f"{delta_f1:+.4f} vs 0.50",
    )

    st.markdown("---")
    st.markdown("### Modelo")
    st.markdown(
        "**Gradient Boosting** tuneado con `GridSearchCV` (5 folds) y "
        "balanceo de clases via `sample_weight`."
    )
    st.code(
        f"n_estimators  = {best_params['n_estimators']}\n"
        f"max_depth     = {best_params['max_depth']}\n"
        f"learning_rate = {best_params['learning_rate']}",
        language="text",
    )
    st.markdown("---")
    st.caption("Universidad San Sebastian · Taller de Aplicaciones · 2026.")


# ============================================================
# HEADER + TABS
# ============================================================
st.title("Clasificador de Ingresos — Adult Census Income")
st.caption(
    "Predice si una persona tiene ingreso anual >USD 50,000 a partir de "
    "13 atributos demograficos y laborales del censo de EE.UU. 1994."
)

tab_rend, tab_prueba, tab_eq, tab_tec = st.tabs([
    "Rendimiento del modelo",
    "Probar el clasificador",
    "Equidad del modelo",
    "Ficha tecnica",
])


# ============================================================
# TAB 1 — RENDIMIENTO (dinamico segun umbral)
# ============================================================
with tab_rend:
    st.subheader(
        f"Metricas en test (n={N_TEST:,}) — umbral "
        f"{curvas['umbral'][idx_u]:.2f}"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",          f"{curvas['accuracy'][idx_u]:.4f}")
    c2.metric("Precision (>50K)",  f"{curvas['precision'][idx_u]:.4f}")
    c3.metric("Recall (>50K)",     f"{curvas['recall'][idx_u]:.4f}")
    c4.metric("F1 (>50K)",         f"{curvas['f1'][idx_u]:.4f}")
    c5.metric("Balanced Acc.",     f"{curvas['balanced_acc'][idx_u]:.4f}")

    idx_max_f1 = int(np.argmax(curvas["f1"]))
    u_optimo = curvas["umbral"][idx_max_f1]
    f1_optimo = curvas["f1"][idx_max_f1]

    if abs(curvas["umbral"][idx_u] - u_optimo) > 0.02:
        st.info(
            f"El F1 maximo se obtiene en umbral **{u_optimo:.2f}** "
            f"(F1 = {f1_optimo:.4f}). Estas en {curvas['umbral'][idx_u]:.2f}. "
            "Prueba moviendo el slider del sidebar."
        )

    st.markdown("---")

    col_cm, col_curvas = st.columns(2)
    with col_cm:
        st.markdown(f"#### Matriz de confusion @ umbral {curvas['umbral'][idx_u]:.2f}")
        cm_din = np.array([
            [curvas["tn"][idx_u], curvas["fp"][idx_u]],
            [curvas["fn"][idx_u], curvas["tp"][idx_u]],
        ])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_din,
            x=["Pred <=50K", "Pred >50K"],
            y=["Real <=50K", "Real >50K"],
            text=cm_din,
            texttemplate="%{text:,}",
            textfont={"size": 18},
            colorscale="Blues",
            showscale=False,
        ))
        fig_cm.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_curvas:
        st.markdown("#### Precision / Recall / F1 vs Umbral")
        df_c = pd.DataFrame({
            "umbral":    curvas["umbral"],
            "Precision": curvas["precision"],
            "Recall":    curvas["recall"],
            "F1":        curvas["f1"],
        }).melt(id_vars="umbral", var_name="Metrica", value_name="Valor")
        fig_curvas = px.line(
            df_c, x="umbral", y="Valor", color="Metrica",
            color_discrete_map={
                "Precision": "#B85042", "Recall": "#5B8DB8", "F1": "#2E8B57"
            },
        )
        fig_curvas.add_vline(
            x=umbral, line_dash="dash", line_color="black",
            annotation_text=f"u={umbral:.2f}", annotation_position="top",
        )
        fig_curvas.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Umbral de decision", yaxis_title="",
        )
        st.plotly_chart(fig_curvas, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Comparacion con otros modelos evaluados (referencia: umbral 0.5)")
    cols_show = ["Accuracy", "Balanced Acc.", "Precision (>50K)",
                 "Recall (>50K)", "F1 (>50K)", "ROC-AUC"]
    st.dataframe(
        tabla_modelos[cols_show].round(4).style
            .highlight_max(axis=0, color="#d4edda")
            .format("{:.4f}", na_rep="—"),
        use_container_width=True,
    )

    st.markdown("#### Curva ROC")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=roc_data["fpr"], y=roc_data["tpr"], mode="lines",
        name=f"modelo_v2 (AUC = {roc_data['auc']:.4f})",
        line=dict(color="#B85042", width=2),
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Aleatorio",
        line=dict(color="grey", width=1, dash="dash"),
    ))
    fig_roc.update_layout(
        height=400, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("#### Variables mas importantes (Gini gain)")
    df_imp = pd.Series(top_features).sort_values(ascending=True).reset_index()
    df_imp.columns = ["Variable", "Importancia"]
    fig_imp = px.bar(
        df_imp, x="Importancia", y="Variable", orientation="h",
        text=df_imp["Importancia"].round(3),
    )
    fig_imp.update_traces(marker_color="#5B8DB8", textposition="outside")
    fig_imp.update_layout(
        height=480, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Importancia (Gini gain)", yaxis_title="",
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# ============================================================
# TAB 2 — PROBAR EL CLASIFICADOR (form + threshold + contribuciones)
# ============================================================
with tab_prueba:
    st.subheader("Ingresa los datos de una persona")
    st.caption(
        f"Umbral de decision actual: **{curvas['umbral'][idx_u]:.2f}** "
        "(modificable en el sidebar). Una probabilidad sobre el umbral se "
        "clasifica como `>50K`."
    )

    with st.form("formulario_prediccion"):
        st.markdown("##### Variables numericas")
        col_n1, col_n2, col_n3 = st.columns(3)
        with col_n1:
            age = st.slider("Edad",
                            int(rangos["age"][0]), int(rangos["age"][1]), value=38)
            education_num = st.slider(
                "Anios de educacion (education-num)",
                int(rangos["education-num"][0]),
                int(rangos["education-num"][1]),
                value=10,
                help="9=HS-grad, 10=Some-college, 13=Bachelors, 14=Masters, 16=Doctorate.",
            )
        with col_n2:
            hours_per_week = st.slider(
                "Horas por semana",
                int(rangos["hours-per-week"][0]),
                int(rangos["hours-per-week"][1]),
                value=40,
            )
            capital_gain = st.number_input(
                "Capital gain (USD anual)",
                min_value=int(rangos["capital-gain"][0]),
                max_value=int(rangos["capital-gain"][1]),
                value=0, step=500,
            )
        with col_n3:
            capital_loss = st.number_input(
                "Capital loss (USD anual)",
                min_value=int(rangos["capital-loss"][0]),
                max_value=int(rangos["capital-loss"][1]),
                value=0, step=100,
            )

        st.markdown("##### Variables categoricas")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            workclass = st.selectbox(
                "Tipo de empleo", categorias["workclass"],
                index=categorias["workclass"].index("Private")
                if "Private" in categorias["workclass"] else 0,
            )
            occupation = st.selectbox(
                "Ocupacion", categorias["occupation"],
                index=categorias["occupation"].index("Prof-specialty")
                if "Prof-specialty" in categorias["occupation"] else 0,
            )
            marital_status = st.selectbox(
                "Estado civil", categorias["marital-status"],
                index=categorias["marital-status"].index("Never-married")
                if "Never-married" in categorias["marital-status"] else 0,
            )
            relationship = st.selectbox(
                "Relacion familiar", categorias["relationship"],
                index=categorias["relationship"].index("Not-in-family")
                if "Not-in-family" in categorias["relationship"] else 0,
            )
        with col_c2:
            sex = st.selectbox(
                "Sexo", categorias["sex"],
                index=categorias["sex"].index("Male")
                if "Male" in categorias["sex"] else 0,
            )
            race = st.selectbox(
                "Raza", categorias["race"],
                index=categorias["race"].index("White")
                if "White" in categorias["race"] else 0,
            )
            native_country = st.selectbox(
                "Pais de origen", categorias["native-country"],
                index=categorias["native-country"].index("United-States")
                if "United-States" in categorias["native-country"] else 0,
            )

        submitted = st.form_submit_button(
            "Predecir ingreso", type="primary", use_container_width=True
        )

    if submitted:
        caso = {
            "age":            age,
            "workclass":      workclass,
            "fnlwgt":         0,
            "education":      "HS-grad",
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
            "income":         "<=50K",
        }

        proba, X_aligned = predecir_proba(caso)
        clase = ">50K" if proba >= umbral else "<=50K"

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
                f"P(>50K) = {proba*100:.1f}%</div>"
                f"<div style='font-size:11px; opacity:0.75; margin-top:4px;'>"
                f"Umbral aplicado: {umbral:.2f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_r2:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": " %", "font": {"size": 36}},
                title={"text": "Probabilidad estimada de ingreso > USD 50,000"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, umbral * 100],   "color": "#f5f5f5"},
                        {"range": [umbral * 100, 100], "color": "#e8f4ea"},
                    ],
                    "threshold": {
                        "line":  {"color": "black", "width": 2},
                        "thickness": 0.85,
                        "value": umbral * 100,
                    },
                },
            ))
            fig_g.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

        # --- Panel de contribuciones por feature ---
        st.markdown("### Que features influyeron mas en esta prediccion")
        st.caption(
            "Cada barra mide la 'fuerza' con la que el valor del usuario en esa "
            "feature empuja la prediccion. **Verde = empuja hacia >50K**; "
            "**rojo = empuja hacia <=50K**. La fuerza es proporcional a la importancia "
            "global de la feature multiplicada por cuanto se aleja el valor del "
            "usuario del punto medio entre las dos clases."
        )

        df_contribs = calcular_contribuciones(X_aligned.iloc[0], top_n=8).iloc[::-1]
        colores_barras = [
            "#2E8B57" if p > 0 else "#B85042" for p in df_contribs["push"]
        ]
        fig_contrib = go.Figure(go.Bar(
            x=df_contribs["push"],
            y=df_contribs["feature"],
            orientation="h",
            marker_color=colores_barras,
            text=[f"valor usuario = {v:.2f}" for v in df_contribs["valor_usuario"]],
            textposition="auto",
            customdata=df_contribs[["media_<=50K", "media_>50K",
                                    "importancia_global"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Valor usuario: %{text}<br>"
                "Media en <=50K: %{customdata[0]:.3f}<br>"
                "Media en >50K: %{customdata[1]:.3f}<br>"
                "Importancia global: %{customdata[2]:.3f}<br>"
                "Fuerza: %{x:.4f}<extra></extra>"
            ),
        ))
        fig_contrib.add_vline(x=0, line_color="black", line_width=1)
        fig_contrib.update_layout(
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Fuerza relativa (positivo empuja a >50K, negativo a <=50K)",
            yaxis_title="",
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

        with st.expander("Ver detalle de los datos ingresados"):
            df_caso = pd.DataFrame.from_dict(
                {k: [v] for k, v in caso.items() if k != "income"},
                orient="columns",
            ).T
            df_caso.columns = ["Valor"]
            st.dataframe(df_caso, use_container_width=True)


# ============================================================
# TAB 3 — EQUIDAD DEL MODELO
# ============================================================
with tab_eq:
    st.subheader("Analisis de equidad por grupos demograficos")
    st.markdown(
        "El dataset Adult refleja el censo de EE.UU. de 1994 y arrastra sesgos "
        "historicos de genero y raza. Esta seccion mide explicitamente como "
        "el modelo se comporta de manera distinta para cada grupo, una "
        "consideracion etica importante en cualquier despliegue real."
    )
    st.caption(
        "Metricas calculadas sobre el test set en el umbral original 0.50. "
        "**Recall** = proporcion de personas con ingreso real >50K que el modelo "
        "identifica correctamente. **Tasa pred+** = proporcion de personas a las "
        "que el modelo asigna >50K (mide demographic parity)."
    )

    # ---- Por sexo ----
    st.markdown("---")
    st.markdown("### Por sexo")
    df_sex = pd.DataFrame.from_dict(fairness["sex"], orient="index")

    col_st, col_sg = st.columns([1, 1])
    with col_st:
        st.dataframe(df_sex.round(4), use_container_width=True)
        if len(df_sex) >= 2:
            gap_recall = df_sex["recall"].max() - df_sex["recall"].min()
            gap_pp = df_sex["tasa_pred_positiva"].max() - df_sex["tasa_pred_positiva"].min()
            cgap1, cgap2 = st.columns(2)
            cgap1.metric(
                "Equal opportunity gap",
                f"{gap_recall*100:.1f} pp",
                help="Diferencia entre el recall mas alto y el mas bajo entre grupos. "
                     "Idealmente cerca de 0.",
            )
            cgap2.metric(
                "Demographic parity gap",
                f"{gap_pp*100:.1f} pp",
                help="Diferencia entre la tasa de >50K asignada al grupo con mas y "
                     "al grupo con menos.",
            )

    with col_sg:
        df_sex_plot = df_sex.reset_index().rename(columns={"index": "Sexo"})
        fig_sex = px.bar(
            df_sex_plot, x="Sexo",
            y=["recall", "precision", "tasa_pred_positiva"],
            barmode="group",
            color_discrete_sequence=["#5B8DB8", "#B85042", "#2E8B57"],
        )
        fig_sex.update_layout(
            height=350, margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Valor", legend_title="Metrica",
        )
        st.plotly_chart(fig_sex, use_container_width=True)

    # ---- Por raza ----
    st.markdown("---")
    st.markdown("### Por raza")
    df_race = pd.DataFrame.from_dict(fairness["race"], orient="index") \
                .sort_values("n", ascending=False)

    col_rt, col_rg = st.columns([1, 1])
    with col_rt:
        st.dataframe(df_race.round(4), use_container_width=True)
        if len(df_race) >= 2:
            gap_recall_r = df_race["recall"].max() - df_race["recall"].min()
            gap_pp_r = df_race["tasa_pred_positiva"].max() - df_race["tasa_pred_positiva"].min()
            cgapr1, cgapr2 = st.columns(2)
            cgapr1.metric("Equal opportunity gap", f"{gap_recall_r*100:.1f} pp")
            cgapr2.metric("Demographic parity gap", f"{gap_pp_r*100:.1f} pp")

    with col_rg:
        df_race_plot = df_race.reset_index().rename(columns={"index": "Raza"})
        fig_race = px.bar(
            df_race_plot, x="Raza",
            y=["recall", "precision", "tasa_pred_positiva"],
            barmode="group",
            color_discrete_sequence=["#5B8DB8", "#B85042", "#2E8B57"],
        )
        fig_race.update_layout(
            height=350, margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Valor", legend_title="Metrica", xaxis_tickangle=-20,
        )
        st.plotly_chart(fig_race, use_container_width=True)

    st.markdown("---")
    st.markdown("### Lectura critica")
    st.markdown(
        "- El modelo predice `>50K` con mucha mayor frecuencia para hombres que "
        "para mujeres. La diferencia refleja la realidad del censo 1994 (los "
        "hombres ganaban en promedio mas) y el modelo absorbe ese patron.\n"
        "- El **recall** tambien difiere: el modelo identifica una proporcion "
        "mayor de hombres con >50K real, pero deja escapar mas casos positivos "
        "entre mujeres.\n"
        "- A nivel de raza, la tasa de prediccion positiva para personas "
        "clasificadas como `Black` es aproximadamente la mitad de la asignada a "
        "personas `White`. Esto reproduce las brechas salariales historicas "
        "presentes en los datos.\n"
        "- Si este modelo se usara para decisiones reales (creditos, contratacion, "
        "subsidios), estos sesgos serian inaceptables y requeririan tecnicas "
        "explicitas de mitigacion: reweighing, adversarial debiasing o "
        "post-processing equalized odds."
    )


# ============================================================
# TAB 4 — FICHA TECNICA
# ============================================================
with tab_tec:
    st.subheader("Ficha tecnica del modelo")

    st.markdown("#### Pipeline de preprocesamiento")
    st.markdown(
        "- Imputacion de `NaN` con `'Unknown'` en `workclass`, `occupation` y `native-country`.\n"
        "- Eliminacion de `fnlwgt` (peso del censo) y `education` (redundante con `education-num`).\n"
        "- Agrupacion de categorias raras y simplificacion de `marital-status` y `native-country`.\n"
        "- Winsorizacion de `capital-gain` (99999 → 41310) y derivadas binarias `has_capital_*`.\n"
        f"- One-hot encoding final: **{len(feat_cols)} features**."
    )

    st.markdown("#### Algoritmo y tuning")
    st.markdown(
        "**`GradientBoostingClassifier`** seleccionado entre 5 candidatos por mejor "
        "F1 y ROC-AUC. Tuneado con `GridSearchCV` (8 combinaciones x 5 folds "
        "estratificados, scoring F1). Balanceo de clases via "
        "`compute_sample_weight('balanced', y_train)`."
    )

    st.markdown("#### Hiperparametros finales")
    st.dataframe(
        pd.DataFrame.from_dict(best_params, orient="index", columns=["Valor"]),
        use_container_width=True,
    )

    st.markdown("#### Mejoras posibles a futuro")
    st.markdown(
        "- XGBoost / LightGBM (suelen superar a Gradient Boosting de sklearn).\n"
        "- Calibracion de probabilidades (`CalibratedClassifierCV`).\n"
        "- Ajuste del umbral segun objetivo de negocio (slider del sidebar lo explora en vivo).\n"
        "- Tecnicas de fairness: reweighing en entrenamiento, equalized odds en post-procesamiento."
    )

    st.markdown("---")
    st.markdown("#### Referencia del dataset")
    st.markdown(
        "Becker, B. & Kohavi, R. (1996). **Adult**. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5XW20"
    )
