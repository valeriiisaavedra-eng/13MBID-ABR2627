# Se importan las librerías necesarias y se suprimen las advertencias
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REGISTERED_MODEL_NAME = "sentimiento-resenas"
ALIAS_PRODUCCION = "champion"
UMBRAL_F1_MACRO = 0.65  # Umbral basado en métrica balanceada

def load_data(data_path: str = "data/processed/datos_integrados.csv"):
    """
    Carga los datos desde un archivo CSV y devuelve un DataFrame de pandas.

    Args:
        data_path (str): Ruta al archivo CSV que contiene los datos procesados.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y: Tupla que contiene los conjuntos de entrenamiento, validación y prueba, así como las características y etiquetas originales.
    """
    # Lectura de los datos
    df = pd.read_csv(data_path)

    target = "sentimiento"
    features_X = df.drop(columns=[target])
    labels_y = df[target]

    # Primero se separa el test final (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_X, labels_y, test_size=0.10, random_state=42, stratify=labels_y
    )

    # Luego se separan train y validation (22% del 90% es aprox. el 20% del total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.22, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y


def create_preprocessor(features_X: pd.DataFrame):
    """
    Crea un preprocesador para los datos de entrada.

    Args:
        features_X (pd.DataFrame): DataFrame que contiene las características de entrada.

    Returns:
        ColumnTransformer: Preprocesador que transforma las características de entrada.
    """
    # Identificación de columnas numéricas y categóricas
    numeric_features = features_X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = features_X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    # SOLUCIÓN TODO: Sub-pipelines con imputadores para robustez en producción
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    text_col = "texto_resena"
    text_transformer = TfidfVectorizer(max_features=300)

    # Creación del preprocesador utilizando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("text", text_transformer, text_col),
        ]
    )

    return preprocessor


def train_model(
    data_path: str = "data/processed/datos_integrados.csv",
    model_output_path: str = "models/trained_model.pkl",
    preprocessor_output_path: str = "models/preprocessor.pkl",
    metrics_output_path: str = "metrics/train_metrics.json",
    figures_output_path: str = "docs/figures",
    registered_model_name: str | None = REGISTERED_MODEL_NAME,
):
    """
    Entrena un modelo de clasificación utilizando los datos procesados y guarda el modelo, el preprocesador y las métricas.

    Args:
        data_path (str): Ruta al archivo CSV que contiene los datos procesados.
        model_output_path (str): Ruta donde se guardará el modelo entrenado.
        preprocessor_output_path (str): Ruta donde se guardará el preprocesador.
        metrics_output_path (str): Ruta donde se guardarán las métricas de entrenamiento.
        figures_output_path (str): Ruta donde se guardarán las figuras generadas durante el entrenamiento.
    """
    # Configuración de las variables de entorno para MLFlow
    # Recordar que debe existir un archivo .env en la raíz del proyecto con las variables de entorno necesarias
    # Las credenciales de MLFlow se pueden obtener desde la interfaz web de DagsHub en la sección de "Settings" y Tokens
    load_dotenv(find_dotenv())
    mlflow.set_experiment(
        "Proyecto 13MBID - V2 - Entrenamiento de modelos de clasificación de sentimientos"
    )

    # Carga de datos
    X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y = load_data(
        data_path
    )

    # Creación del preprocesador
    preprocessor = create_preprocessor(features_X)

    # SOLUCIÓN TODO: Definición de múltiples candidatos para selección basada en F1-Macro
    candidatos = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    }

    mejor_score = -1
    mejor_nombre = None
    mejor_pipeline = None
    mejores_metricas = {}

    # Evaluación y selección del mejor modelo basado en F1-Score Macro
    for nombre, modelo in candidatos.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", modelo)])
        pipe.fit(X_train, y_train)
        
        y_val_pred = pipe.predict(X_val)
        score_val = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        
        print(f"Candidato [{nombre}] - Validation F1-Macro: {score_val:.4f}")
        
        if score_val > mejor_score:
            mejor_score = score_val
            mejor_nombre = nombre
            mejor_pipeline = pipe

    print(f"\n>>> MEJOR MODELO SELECCIONADO: {mejor_nombre} con F1-Macro de {mejor_score:.4f}\n")
   
    # Evaluación final del mejor modelo en el conjunto de test
    pipeline = mejor_pipeline
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)

    # Cálculo de métricas de evaluación
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision_macro": precision_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "test_recall_macro": recall_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "test_f1_macro": f1_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "test_recall_negative": recall_score(
            y_test, y_test_pred, pos_label="negativo", average="macro", zero_division=0
        ),
        "test_roc_aux_ovr": roc_auc_score(
            y_test, y_test_proba, multi_class="ovr", labels=pipeline.classes_
        ),
    }

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Guardadado de la matriz de confusión
    Path(figures_output_path).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_test_pred, labels=pipeline.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Logistic Regression - Test Set")
    cm_fig_path = f"{figures_output_path}/confusion_matrix_test.png"
    plt.savefig(cm_fig_path)
    plt.close()

    # Registro general en MLFlow
    signature = infer_signature(X_train, pipeline.predict(X_train))

    with mlflow.start_run(run_name=f"Pipeline (prod) - {mejor_nombre}") as run:
        mlflow.log_param("selected_model_algorithm", mejor_nombre)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_fig_path, artifact_path="figures")
        model_info = mlflow.sklearn.log_model(
            pipeline, "model", signature=signature, input_example=X_train.head(3),
            registered_model_name=registered_model_name, skops_trusted_types=["numpy.dtype"],
        )

        run_id = mlflow.active_run().info.run_id
        print(f"Modelo registrado en MLFlow con run ID: {run_id}")


    if registered_model_name:
        version = model_info.registered_model_version
        if metrics["test_f1_macro"] >= UMBRAL_F1_MACRO:
            mlflow.MlflowClient().set_registered_model_alias(
                name=registered_model_name, version=version, alias=ALIAS_PRODUCCION
            )
            print(f"Modelo registrado como '{registered_model_name}' con alias '{ALIAS_PRODUCCION}'")
        else:
            print(f"Modelo no cumple con el umbral de accuracy ({UMBRAL_F1_MACRO}). No se registra como '{ALIAS_PRODUCCION}'.")
             
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(preprocessor_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Guardado del modelo entrenado y del preprocesador
    joblib.dump(pipeline, model_output_path)
    joblib.dump(pipeline.named_steps["prep"], preprocessor_output_path)

    # Guardado de métricas en un archivo JSON
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return pipeline, pipeline.named_steps["prep"], metrics


if __name__ == "__main__":
    train_model()