import pandas as pd
import pytest
from pandera.pandas import DataFrameSchema, Column, Check


GENEROS =["Femenino", "Masculino", "Otro"]
SENTIMIENTOS =["Positivo", "Negativo", "Neutral"]
CANALES_COMPRA = ["Online", "Offline"]

@pytest.fixture
def datos_clientes():
    """ Función para cargar los datos de clientes a un dataframe
    
    Args:
        None

    Returns:
        pd.Dataframe: Dataframe con los datos de clientes
    """
    df = pd.read_csv("data/raw/datos_clientes.csv", sep=";")
    return df


@pytest.fixture
def datos_resenas():
    """ Función para cargar los datos de reseñas a un dataframe
   
    Args:
        None

    Returns:
        pd.Dataframe: Dataframe con los datos de reseñas
    """
    df = pd.read_csv("data/raw/datos_resenas.csv", sep=";")
    return df


def test_schema_datos_clientes(datos_clientes):
    """ Funcion para validar el esquema de los datos de clientes

    Args:
        datos_clientes (pd.Dataframe): Dataframe con los datos de clientes
    
    Returns:
        None
    """
    schema = DataFrameSchema(
        {
            "id_cliente": Column(int, nullable=False), 
            "edad": Column(float, nullable=True, checks=Check.in_range(18, 90)),
            "genero": Column(str, nullable=False, checks=Check.isin(GENEROS)),
            "ciudad": Column(str, nullable=False),
            "antiguedad_cliente_meses": Column(int, nullable=True, checks=Check.ge(0)),
            "canal_registro": Column(str, nullable=False),
        },
        drop_invalid_rows=True #### para descartar o ignorar automáticamente las filas incorrectas en lugar de hacer fallar la prueba
    )
    # Lazy = True permite que se validen todas las columnas y se muestren todos los
    # errores encontrados en lugar de detenerse en el primer error.
    schema.validate(datos_clientes, lazy=True)


def test_schema_datos_resenas(datos_resenas):
    """ Funcion para validar el esquema de los datos de reseñas

    Args:
        datos_resenas (pd.Dataframe): Dataframe con los datos de reseñas
    
    Returns:
        None
    """
    schema = DataFrameSchema(
        {
            "id_resena": Column(int, nullable=False), 
            "id_cliente": Column(int, nullable=False), 
            "fecha_resena": Column(str, nullable=True), 
            "calificacion": Column(int, nullable=False, checks=Check.in_range(1, 5)),
            "sentimiento": Column(str, nullable=False, checks=Check.isin(SENTIMIENTOS)),
            "categoria_producto": Column(str, nullable=False),
            "canal_compra": Column(str, nullable=False, checks=Check.isin(CANALES_COMPRA)),
            "plataforma": Column(str, nullable=False),
            "texto_resena": Column(str, nullable=False),
            "tiempo_respuesta_horas": Column(float, nullable=True, checks=Check.ge(0)),
            "problema_resuelto": Column(str, nullable=False),
            "reclamo_formal": Column(str, nullable=False),
            "importe_compra": Column(float, nullable=False, checks=Check.ge(0)),
        
        },
        drop_invalid_rows=True
    )
    schema.validate(datos_resenas, lazy=True)


def test_basicos_clientes(datos_clientes):
    """ Función para validar los datos básicos de los clientes

    Args:
        datos_clientes (pd.Dataframe): Dataframe con los datos de clientes
    
    Returns:
        None
    """
    # Validar que no existan valores nulos en las columnas obligatorias
    assert not datos_clientes.empty, "El dataframe de clientes está vacío"
    assert datos_clientes.shape[1] == 6, "El dataframe de clientes no tiene 6 columnas"
    assert datos_clientes["id_cliente"].notnull().all(), "La columna id_cliente tiene valores nulos"
    # EXTRA: pueden completar otras validaciones básicas

def test_basicos_resenas(datos_resenas):
    """ Función para validar los datos básicos de las reseñas

    Args:
        datos_resenas (pd.Dataframe): Dataframe con los datos de reseñas
    
    Returns:
        None
    """
    # Validar que no existan valores nulos en las columnas obligatorias
    assert not datos_resenas.empty, "El dataframe de reseñas está vacío"
    assert datos_resenas.shape[1] == 13, "El dataframe de clientes no tiene 13 columnas"
    assert datos_resenas["id_resena"].notnull().all(), "La columna id_resena tiene valores nulos"
    assert datos_resenas["id_cliente"].notnull().all(), "La columna id_cliente tiene valores nulos"
    # EXTRA: pueden completar otras validaciones básicas


def test_integridad_referencial(datos_clientes, datos_resenas):
    """ Función para validar la integridad referencial entre los datos de clientes y reseñas

    Args:
        datos_clientes (pd.Dataframe): Dataframe con los datos de clientes
        datos_resenas (pd.Dataframe): Dataframe con los datos de reseñas
    
    Returns:
        None
    """
    # Validar que todos los id-cliente en datos_resenas existan en datos_clientes
    df_ids= datos_resenas [['id_cliente']].merge(
        datos_clientes[['id_cliente']],
        on='id_cliente',
        how='left',
        indicator=True # generar la columna _merge con el resultado de la unión
    )
    integridad_referencial = DataFrameSchema(
        {
            "_merge": Column(str, nullable=False, checks=Check.isin(["both"]))
        }
    )
    integridad_referencial.validate(df_ids, lazy=True)


##############################################################################
# SOLUCIÓN TODO: Se agrega test unicidad de claves (id) y la validación de
# reglas de negocio
##############################################################################

def test_unicidad_claves(datos_clientes, datos_resenas):
    """ Función para validar la unicidad de las llaves primarias en ambos datasets
    """
    # Validar que no existan duplicados en id_cliente
    assert datos_clientes['id_cliente'].is_unique, "Existen id_cliente duplicados en el dataset de clientes"
    
    # Validar que no existan duplicados en id_resena
    assert datos_resenas['id_resena'].is_unique, "Existen id_resena duplicados en el dataset de reseñas"


def test_regla_negocio_calificacion_sentimiento(datos_resenas):
    """ EXTRA: Validar la consistencia lógica entre 'calificacion' y 'sentimiento'.
        Regla: Si la calificación es baja (1 o 2), la proporción de sentimientos 
        positivos no debe superar el 5%.
    """
    # Filtrar calificaciones bajas
    calificaciones_bajas = datos_resenas[datos_resenas['calificacion'] <= 2]
    
    if len(calificaciones_bajas) > 0:
        # Calcular porcentaje de sentimientos positivos en calificaciones bajas
        positivos_en_bajas = (calificaciones_bajas['sentimiento'] == 'Positivo').mean()
        
        assert positivos_en_bajas <= 0.05, (
            f"Violación de regla de negocio: El {positivos_en_bajas*100:.2f}% de las "
            f"reseñas con calificación baja (<=2) tienen sentimiento 'Positivo' (Máx permitido: 5%)."
        )