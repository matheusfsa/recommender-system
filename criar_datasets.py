from imdb import IMDb
import numpy as np
import re
import pandas as pd

def has_name(a):
    return True if a.data else False

def get_filme(ia, movieId, imdbId):
    movie = ia.get_movie(imdbId)
    res = dict()
    res['movieId'] = movieId
    try:
        res['title'] = movie['title']
    except KeyError:
        res['title'] = None
    try:
        res['genres'] = '|'.join(movie['genres'])
    except KeyError:
        res['genres'] = None
    try:
        res['year'] = movie['year']
    except KeyError:
        res['year'] = None
    try:
        res['cast'] = '|'.join([a.data['name'] for a in filter(has_name, movie['cast'])])
    except KeyError:
        res['cast'] = None
    try:
        res['directors'] = '|'.join([a.data['name'] for a in filter(has_name, movie['directors'])])
    except KeyError:
        res['directors'] = None
    try:
        res['writers'] = '|'.join([a.data['name'] if a.data else '' for a in filter(has_name, movie['writers'])])
    except KeyError:
        res['writers'] = None
    try:
        res['producers'] = '|'.join([a.data['name'] for a in filter(has_name,movie['producers'])])
    except KeyError:
        res['producers'] = None
    try:
        res['synopsis'] = movie['synopsis'][0]
    except KeyError:
        res['synopsis'] = None
    try:
       res['runtimes'] = np.mean([int(re.sub("^[a-zA-Z]+:", '', x)) for x in movie['runtimes']])
    except KeyError:
        res['runtimes'] = None
    try:
        res['color'] = '|'.join(movie['color'])
    except KeyError:
        res['color'] = None
    try:
        res['rating'] = float(movie['rating'])
    except KeyError:
        res['rating'] = None
    try:
        res['plots'] = '|'.join([re.sub("::.+", '', x) for x in movie['plot']])
    except KeyError:
        res['plots'] = None
    return res

def criar_dataset():
    df_links = pd.read_csv('./datasets/ml-20m/links.csv', dtype=str)
    df_movies = pd.DataFrame()
    ia = IMDb()
    for a in df_links.iterrows():
        m = get_filme(ia, a[1][0], a[1][1])
        df_movies = df_movies.append(m, ignore_index=True)
        print('Filme adicionado:', m['title'])
    df_movies.to_csv('movies_imdb.csv', index=False)
    return df_movies