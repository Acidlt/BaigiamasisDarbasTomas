import pandas as pd
import numpy as np
import sqlite3
import time
from sklearn.neighbors import NearestNeighbors
from website.search import search_title, clean_title
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import average_precision_score,mean_squared_error, mean_absolute_error

# Load your movie data, ratings data, and preprocess as needed
conn = sqlite3.connect(r'C:\Users\ttzur\Desktop\WebWithFlask\instance\database.db')
    # get all movies from the database
sql_query = pd.read_sql_query ('''
                               SELECT
                               *
                               FROM movie
                               ''', conn)

movies = pd.DataFrame(sql_query, columns = ['id', 'title','genre'])
movies = movies.rename(columns={'id': 'movie_id'})
movies["title"] = movies["title"].apply(clean_title)
sql_query = pd.read_sql_query ('''
                               SELECT
                               *
                               FROM ratings
                               ''', conn)

ratings = pd.DataFrame(sql_query, columns = ['id','user_id','movie_id', 'rating'])

sql_query = pd.read_sql_query ('''
                               SELECT
                               *
                               FROM link
                               ''', conn)

links = pd.DataFrame(sql_query, columns = ['id','movie_id','imdb_id', 'tmdb_id'])
links['imdb_id'] = links['imdb_id'].fillna('')
links['tmdb_id'] = links['tmdb_id'].fillna('')


movie_user_matrix = ratings.pivot_table(index='movie_id', columns='user_id', values='rating').fillna(0)


normalized_matrix = movie_user_matrix - movie_user_matrix.mean(axis=1).values.reshape(-1, 1)


k = 50  
knn_model = NearestNeighbors(n_neighbors=k, metric='cosine')  
knn_model.fit(normalized_matrix)




def recommend_movies_knn(title, num_recommendations=10):
    movieTitleSearch=search_title(title)
    movie_id=movieTitleSearch.iloc[0]["movie_id"]
    distances, indices = knn_model.kneighbors(normalized_matrix.loc[movie_id].values.reshape(1, -1))
    recommended_movie_ids = [movie_user_matrix.index[i] for i in indices.flatten()][1:]
    similarity_scores = 1 - distances.flatten()
    # Filter the recommended movies and return the top num_recommendations
    recommended_movies = movies[movies['movie_id'].isin(recommended_movie_ids)].head(num_recommendations)
    result = recommended_movies.merge(links, on='movie_id',how='left')[["movie_id", "title","genre", "imdb_id", "tmdb_id"]]
    result["title"] = result["title"].str.title()
    return result, movieTitleSearch, similarity_scores[1:]




def recommend_movies_knn_User(user_id, num_recommendations=50):
    # Get the user's movie ratings
    user_ratings = ratings[ratings['user_id'] == user_id]

    # Get the user's preference vector from the normalized_matrix
    user_preferences = normalized_matrix.loc[user_ratings['movie_id']].mean(axis=0).values.reshape(1, -1)
     
    
    # Get the k nearest neighbors to the user's preference vector using the k-NN model
    distances, indices = knn_model.kneighbors(user_preferences)

    # Calculate the weighted average of the neighbors' ratings for each movie
    sim_scores = 1 - distances.flatten()
    movie_sim_matrix = normalized_matrix.iloc[indices.flatten()].copy()  # Get the neighbors' movie ratings
    
    # Transpose the movie_sim_matrix before adding the similarity column
    movie_sim_matrix = movie_sim_matrix.T

    # Add the similarity column to the transposed movie_sim_matrix
    similarity_series = pd.Series(sim_scores, index=movie_sim_matrix.columns, name='similarity')
    movie_sim_matrix = movie_sim_matrix.append(similarity_series)

    # Calculate the weighted average of the neighbors' ratings for each movie
    weighted_movie_ratings = movie_sim_matrix.apply(lambda row: row * row['similarity'], axis=0).sum() / sim_scores.sum()
    
    
    # Sort the movies by their predicted ratings and exclude already rated movies
    predicted_ratings = weighted_movie_ratings.sort_values(ascending=False)
    
    #recommended_movie_ids = [movie_id for movie_id in predicted_ratings.index if movie_id not in rated_movie_ids]
    recommended_movie_ids = [movie_id for movie_id in predicted_ratings.index] 
    # Select the top num_recommendations
    top_movies_indices = recommended_movie_ids[:num_recommendations]

    # Get the movie details for the recommended movies
    recommended_movies = movies[movies['movie_id'].isin(top_movies_indices)]
    recommended_movies_hybrid = recommended_movies.merge(links, on='movie_id', how='left')[["movie_id", "title", "genre", "imdb_id", "tmdb_id"]]
    result=recommended_movies_hybrid.head(10)
    result["title"] = result["title"].str.title()
    weighted_movie_ratings = weighted_movie_ratings[:num_recommendations]

    return result, weighted_movie_ratings,recommended_movies_hybrid

# # Split the dataset into train and test sets
# train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
# average_user_ratings = ratings.groupby('user_id')['rating'].mean()
# # Predict the ratings for the test set
# predicted_ratings = []
# for user_id, movie_id, true_rating in tqdm(test_data[['user_id', 'movie_id', 'rating']].values):
#     recommended_movies, user_predicted_ratings, _ = recommend_movies_knn_User(user_id)
    
#     #predicted_rating = user_predicted_ratings.get(movie_id, 0)  # Get the predicted rating for the movie, default to 0 if not found
#     predicted_rating = user_predicted_ratings.get(movie_id, average_user_ratings[user_id])
#     predicted_ratings.append(predicted_rating)

# # Calculate RMSE
# rmse = np.sqrt(mean_squared_error(test_data['rating'], predicted_ratings))
# print("RMSE:", rmse)



#========================Time Test=============================================================================
# user_id=2
# start_time = time.time()
# recommend_movies_knn_User(user_id)
# knn_time = time.time() - start_time
# print("k-NN Time: {:.2f} seconds".format(knn_time))



