import numpy as np
import pandas as pd
import sqlite3
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz


def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title.lower()

def search_title(title):
    conn = sqlite3.connect(r'C:\Users\ttzur\Desktop\WebWithFlask\instance\database.db')
    sql_query = pd.read_sql_query ('''
                               SELECT
                               *
                               FROM movie
                               ''', conn)
    movies = pd.DataFrame(sql_query, columns = ['id', 'title', 'genre'])
    movies = movies.rename(columns={'id': 'movie_id'})
    sql_query = pd.read_sql_query ('''
                               SELECT
                               *
                               FROM link
                               ''', conn)
    links = pd.DataFrame(sql_query, columns = ['id','movie_id','imdb_id', 'tmdb_id'])
    links['imdb_id'] = links['imdb_id'].fillna('')
    links['tmdb_id'] = links['tmdb_id'].fillna('')
    
    movies["title"] = movies["title"].apply(clean_title)
    title = clean_title(title)

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(movies["title"])

    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]

    result = movies.iloc[indices][::-1]
    
    # Use fuzzy matching to further refine the results
    result["similarity"] = result["title"].apply(lambda x: fuzz.token_set_ratio(title, x))
    result = result.sort_values(by="similarity", ascending=False)
    result = result.merge(links, on="movie_id")[["movie_id", "title","genre", "imdb_id", "tmdb_id"]]
    result["title"] = result["title"].str.title()
    result = result.head(5)
    return result
  