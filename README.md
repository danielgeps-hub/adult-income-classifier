# Adult Income Classifier — Streamlit App

Publicación del modelo de clasificación desarrollado en la **Solemne 1** del
Taller de Aplicaciones, sobre el dataset **Adult / Census Income** del
UCI Machine Learning Repository.

## Modelo

`GradientBoostingClassifier` tuneado con `GridSearchCV` (8 combinaciones × 5
folds estratificados) y balanceo de clases vía `sample_weight`.

| Métrica          | Test set (16.281 muestras) |
|------------------|----------------------------|
| Accuracy         | 0.8358                     |
| Precision (>50K) | 0.6079                     |
| Recall (>50K)    | 0.8586                     |
| **F1 (>50K)**    | **0.7118**                 |
| **ROC-AUC**      | **0.9277**                 |

## Estructura

```
streamlit_app/
├── app.py                  # Aplicación Streamlit
├── modelo_v2_bundle.pkl    # Modelo entrenado + metadatos
├── requirements.txt        # Dependencias
└── README.md
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

La app se abre en `http://localhost:8501`.

## Despliegue en Streamlit Community Cloud

1. Crear un repositorio público en GitHub con el contenido de esta carpeta.
2. Ir a [share.streamlit.io](https://share.streamlit.io), conectar GitHub.
3. *New app* → seleccionar el repo, branch `main`, archivo `app.py`.
4. Deploy. El link queda permanente (`https://<usuario>-<repo>.streamlit.app`).

## Funcionalidades

- **Rendimiento del modelo**: KPIs, matriz de confusión, curva ROC, comparación con 5 modelos baseline e importancia de variables.
- **Probar el clasificador**: formulario interactivo (sliders + selectores) que permite ingresar los datos de una persona y obtener la probabilidad estimada de ingreso > USD 50.000.
- **Ficha técnica**: pipeline de preprocesamiento, hiperparámetros, mejoras futuras y referencia del dataset.

## Autor

Daniel — Universidad San Sebastián · Taller de Aplicaciones 2026.
