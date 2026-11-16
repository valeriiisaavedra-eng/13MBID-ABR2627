import pandas as pd
from pathlib import Path
import json

def create_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea atributos adicionales en el DataFrame de entrada.

    Args:
        df (pd.DataFrame): DataFrame de entrada con los datos originales.

    Returns:
        pd.DataFrame: DataFrame con los atributos adicionales.
    """
    ####################################################################
    # Creación de atributos nuevos
    #################################################################### 
    # Longitud del texto de la reseña, en caracteres
    df["longitud_resena"] = df["texto_resena"].str.len()
    # Cantidad de palabras del texto de la reseña
    df["cantidad_palabras_resena"] = df["texto_resena"].str.split().apply(len)
    # Tiempo de respuesta expresado en días, más interpretable que en horas
    df["tiempo_respuesta_dias"] = df["tiempo_respuesta_horas"] / 24
    return df


def process_data(
    datos_clientes: str = "data/raw/datos_clientes.csv",
    datos_resenas: str = "data/raw/datos_resenas.csv",
    output_dir: str = "data/processed",
) -> None:
    """ Función para procesar los datos de clientes y reseñas

    Args:
        datos_clientes (str): Ruta al archivo CSV con los datos de clientes
        datos_resenas (str): Ruta al archivo CSV con los datos de reseñas
        output_dir (str): Ruta al directorio donde se guardarán los archivos procesados

    Returns:
        None
    """
    # Lectura de los datos
    df_clientes = pd.read_csv(datos_clientes, sep=";")
    df_resenas = pd.read_csv(datos_resenas, sep=";")

    # Declaración de variables iniciales
    n_filas_clientes_inicial = len(df_clientes)
    n_filas_resenas_inicial = len(df_resenas)
    nulos_clientes_inicial = df_clientes.isnull().sum().to_dict()
    nulos_resenas_inicial = df_resenas.isnull().sum().to_dict()

    ####################################################################
    # Selección de los datos (filtro de columnas)
    # Agregar si corresponde otras acciones en función de la evaluación
    # de calidad de los datos del paso previo.
    ####################################################################
    col_eliminar_clientes = ['canal_registro']
    df_clientes.drop(col_eliminar_clientes, inplace=True, axis=1)

    ####################################################################
    # Limpieza de los datos (filtro de filas)
    # Agregar si corresponde otras acciones en función de la evaluación
    # de calidad de los datos del paso previo.
    ####################################################################
    df_clientes_filtrado = df_clientes[(df_clientes['edad'] < 90) | (df_clientes['edad'].isna())]
    df_resenas_filtrado = df_resenas.drop_duplicates().copy()

    duplicados_resenas_eliminados = n_filas_resenas_inicial - len(df_resenas_filtrado)
    
    ####################################################################
    # Tratamiento de valores nulos
    #################################################################### 
    df_clientes_filtrado.loc[:,'edad'] = df_clientes_filtrado.groupby("ciudad")["edad"]\
                        .transform(lambda x: x.fillna(x.median()))
    df_resenas_filtrado['tiempo_respuesta_horas'] = df_resenas_filtrado.groupby("categoria_producto")["tiempo_respuesta_horas"]\
                        .transform(lambda x: x.fillna(x.median()))

    ####################################################################
    # Integración de los datos
    #################################################################### 
    df_integrado = pd.merge(
        df_resenas_filtrado,
        df_clientes_filtrado, 
        on='id_cliente', 
        how='inner'
    )

    ####################################################################
    # Creación de atributos nuevos
    #################################################################### 
    # Longitud del texto de la reseña, en caracteres
    df_integrado["longitud_resena"] = df_integrado["texto_resena"].str.len()
    # Cantidad de palabras del texto de la reseña
    df_integrado["cantidad_palabras_resena"] = df_integrado["texto_resena"].str.split().apply(len)
    # Tiempo de respuesta expresado en días, más interpretable que en horas
    df_integrado["tiempo_respuesta_dias"] = df_integrado["tiempo_respuesta_horas"] / 24

    ####################################################################
    # Formateo de los datos
    #################################################################### 
    # Se eliminan columnas identificadoras, la fecha (no se utiliza en esta etapa)
    # y 'calificacion', dado que determina directamente a 'sentimiento' y generaría data leakage
    columnas_a_eliminar = [
        "id_cliente",
        "id_resena",
        "calificacion",
        "fecha_resena",
    ]

    df_final = df_integrado.drop(columns=columnas_a_eliminar)

    ####################################################################
    # Generar el csv de salida
    #################################################################### 
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_final.to_csv(f"{output_dir}/datos_integrados.csv", index=False)


    ####################################################################
    # SOLUCIÓN TODO: otros atributos que pueden generarse
    ####################################################################
    # 1. Cantidad de reseñas por cliente
    df_integrado["cantidad_resenas_cliente"] = df_integrado.groupby("id_cliente")["id_resena"].transform("count")
    # 2. Gasto promedio por cliente
    df_integrado["gasto_promedio_cliente"] = df_integrado.groupby("id_cliente")["importe_compra"].transform("mean")

    ####################################################################
    # Formateo de los datos
    #################################################################### 
    columnas_a_eliminar = [
        "id_cliente",
        "id_resena",
        "calificacion",
        "fecha_resena",
    ]

    df_final = df_integrado.drop(columns=columnas_a_eliminar)

    ####################################################################
    # Generar el csv de salida
    #################################################################### 
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_final.to_csv(f"{output_dir}/datos_integrados.csv", index=False)

    #################################################################### 
    # EXTRA: Generar un reporte de las transformaciones aplicadas (JSON)
    #################################################################### 
    reporte_transformaciones = {
        "filas_iniciales_clientes": n_filas_clientes_inicial,
        "filas_iniciales_resenas": n_filas_resenas_inicial,
        "filas_clientes_filtrados": len(df_clientes_filtrado),
        "filas_resenas_filtradas": len(df_resenas_filtrado),
        "duplicados_resenas_eliminados": duplicados_resenas_eliminados,
        "filas_finales_integradas": len(df_final),
        "columnas_finales": list(df_final.columns),
        "valores_nulos_iniciales_clientes": nulos_clientes_inicial,
        "valores_nulos_iniciales_resenas": nulos_resenas_inicial,
        "valores_nulos_finales": df_final.isnull().sum().to_dict()
    }

    reporte_path = Path(output_dir) / "reporte_transformaciones.json"
    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(reporte_transformaciones, f, ensure_ascii=False, indent=4)
     


if __name__ == "__main__":
    process_data()