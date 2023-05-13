from fastapi import FastAPI


app = FastAPI()

#http://127.0.0.1:8000
class peliculas 

@app.get('/peliculas_mes/{mes}')
def peliculas_mes(mes:str):
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes historicamente'''
    return {'mes':mes, 'cantidad':respuesta}

@app.get('/peliculas_dis/{dis}')
def peliculas_dia(dia:str):
    '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrebaron ese dia historicamente'''
    return {'dia':dia, 'cantidad':respuesta}

@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''
    return {'franquicia':franquicia, 'cantidad':respuesta, 'ganancia_total':respuesta, 'ganancia_promedio':respuesta}

@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo'''
    return {'pais':pais, 'cantidad':respuesta}

@app.get('/productoras/{productora}')
def productoras(productora:str):
    '''Ingresas la productora, retornando la ganancia toal y la cantidad de peliculas que produjeron'''
    return {'productora':productora, 'ganancia_total':respuesta, 'cantidad':respuesta}

@app.get('/retorno/{pelicula}')
def retorno(pelicula:str):
    '''Ingresas la pelicula, retornando la inversion, la ganancia, el retorno y el año en el que se lanzo'''
    return {'pelicula':pelicula, 'inversion':respuesta, 'ganacia':respuesta,'retorno':respuesta, 'anio':respuesta}

# ML
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    return {'lista recomendada': respuesta}