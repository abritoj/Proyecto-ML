from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


app = FastAPI()


movies=pd.read_csv('df_definitivo.csv')
movies['release_date'] = pd.to_datetime(movies['release_date'])

@app.get('/peliculas_mes/{mes}')
def peliculas_mes(mes:str):
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes historicamente'''
    # Diccionario para convertir el nombre del mes en español a su número
    meses = {'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
             'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12}
    # Convierte el nombre del mes en español a su número
    mes_numero = meses.get(mes.lower())
    # Verifica si el mes es válido
    if mes_numero is None:
        return f"Error: Mes '{mes}' no válido"
    # Filtra el DataFrame por el mes especificado
    filtro = movies[movies['release_date'].dt.month == mes_numero]
    # Cuenta el número de películas
    count = filtro['title'].count()
    # Devuelve el resultado en el formato especificado
    return {'mes':mes, 'cantidad':str(count)}

@app.get('/peliculas_dis/{dis}')
def peliculas_dia(dia:str):
    '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrebaron ese dia historicamente'''
   # Crea una lista con los días de la semana en español
    dias = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
    # Encuentra el índice del día especificado en la lista de días
    dia_index = dias.index(dia.lower())
    # Crea una columna con el día de la semana en el DataFrame
    movies['dia_semana'] = movies['release_date'].dt.dayofweek
    # Filtra el DataFrame por el día especificado
    filtro = movies[movies['dia_semana'] == dia_index]
    # Cuenta el número total de películas
    count = filtro['title'].count()
    # Devuelve el resultado en el formato especificado

    return {'dia':dia, 'cantidad':str(count)}

@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''
    # Filtra el DataFrame por la franquicia especificada
    filtro = movies[movies['belongs_to_collection'] == franquicia]
    # Cuenta el número de películas
    cantidad = filtro['title'].count()
    # Calcula la ganancia total
    ganancia_total = filtro['revenue'].sum()
    # Calcula la ganancia promedio
    ganancia_promedio = filtro['revenue'].mean()
    # Devuelve el resultado en el formato especificado
    result = {'franquicia': franquicia,
              'cantidad': cantidad,
              'ganancia_total': ganancia_total,
              'ganancia_promedio': ganancia_promedio}
    return str(result)
@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    # Contar la frecuencia de los valores en la columna 'name_countrie'
    paises = movies['name_countrie'].value_counts()
    
    # Obtener la cantidad de películas producidas en el país especificado
    respuesta = paises.get(pais, 0)
    
    # Devolver el resultado
    return {'pais': pais, 'cantidad':str(respuesta)}

    

@app.get('/productoras/{productora}')
def productoras(productora:str):
    # Agrupar los datos por la columna 'production_companies'
    grupos = movies.groupby('production_companies')
    
    # Obtener el grupo correspondiente a la productora especificada
    grupo = grupos.get_group(productora)
    
    # Calcular la ganancia total y la cantidad de películas producidas por la productora
    ganancia_total = grupo['revenue'].sum()
    cantidad = len(grupo)
    
    # Devolver el resultado
    return {'productora': productora, 'ganancia_total': str(ganancia_total), 'cantidad': str(cantidad)}

@app.get('/retorno/{pelicula}')
def retorno(pelicula:str):
    '''Ingresas la pelicula, retornando la inversion, la ganancia, el retorno y el año en el que se lanzo'''
    mask = movies['title'] == pelicula
    inversion = movies[mask]['budget'].sum()
    ganancia = movies[mask]['revenue'].sum()
    retorno = (ganancia - inversion) / inversion if inversion != 0 else 0
    anio = movies[mask]['release_year'].sum()
    return {'pelicula':pelicula, 'inversion':inversion, 'ganancia':ganancia,'retorno':retorno, 'anio':anio}

# ML
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    # Eliminar los valores NaN de las columnas 'title' y 'genres'
    df = movies.dropna(subset=['title', 'genres'])
    
    # Crear un vectorizador TF-IDF para extraer características de los datos textuales
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Representar cada película como un vector numérico
    matriz_tfidf = tfidf.fit_transform(df['title'] + ' ' + df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else ''))
    
    # Crear un modelo de vecinos más cercanos
    modelo_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    
    # Entrenar el modelo con los datos del DataFrame
    modelo_knn.fit(matriz_tfidf)
    
    # Buscar la película especificada en el DataFrame
    pelicula = df[df['title'] == titulo].index[0]
    
    # Encontrar las 6 películas más similares a la película especificada
    distancias, indices = modelo_knn.kneighbors(matriz_tfidf[pelicula], n_neighbors=6)
    
    # Seleccionar las 5 películas más similares (excluyendo la película especificada)
    recomendadas = df.iloc[indices[0][1:]]['title'].tolist()
    
    # Devolver el resultado
    return {'lista recomendada': recomendadas}

    