from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


app = FastAPI()


movies = pd.read_csv('movies_dataset_limpio.csv')


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
    count = len(filtro['title'].count())
    # Devuelve el resultado en el formato especificado

    return {'mes':mes, 'cantidad':count}

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
    count = len(filtro['title'].count())
    # Devuelve el resultado en el formato especificado

    return {'dia':dia, 'cantidad':count}

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
    return {'franquicia':franquicia, 'cantidad':cantidad, 'ganancia_total':ganancia_total, 'ganancia_promedio':ganancia_promedio}

@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo'''
    cantidad = sum(movies['production_countries'].apply(lambda x: pais in x))


    return {'pais':pais, 'cantidad':cantidad}

@app.get('/productoras/{productora}')
def productoras(productora:str):
    '''Ingresas la productora, retornando la ganancia toal y la cantidad de peliculas que produjeron'''
     # Filtra el DataFrame por la productora especificada
    filtro = movies[movies['production_companies'].apply(lambda x: productora in x)]
    # Cuenta el número de películas
    cantidad = filtro['title'].count()
    # Calcula la ganancia total
    ganancia_total = filtro['revenue'].sum()
    # Devuelve el resultado en el formato especificado
    result = {'productora': productora,
              'cantidad': cantidad,
              'ganancia_total': ganancia_total}
    return {'productora':productora, 'ganancia_total':ganancia_total, 'cantidad':cantidad}

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
genres = movies['genres']
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(genres)

production_companies = movies['production_companies']
mlb = MultiLabelBinarizer()
production_companies_encoded = mlb.fit_transform(production_companies)

# Luego,  concatenar las matrices para obtener una matriz de características para cada película
features = np.concatenate([genres_encoded, production_companies_encoded], axis=1)

# Luego, calcular la similitud entre todas las películas utilizando la distancia del coseno
similarity_matrix = cosine_similarity(features)
@app.get('/recomendacion/{titulo}')
def recomendacion(title:str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
      # Obtenemos el índice de la película en el dataframe
    idx = movies[movies['title'] == title].index[0]
    # Obtenemos las similitudes con todas las demás películas
    similarities = similarity_matrix[idx]
    # Ordenamos las películas por similitud y seleccionamos las 5 más similares
    similar_movies_idx = np.argsort(similarities)[::-1][1:6]
    similar_movies = movies.iloc[similar_movies_idx]['title']
    return similar_movies.tolist()

    