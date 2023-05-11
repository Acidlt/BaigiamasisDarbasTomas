from website.knn import recommend_movies_knn, recommend_movies_knn_User
from website.cosine import recommend_movies_cosine, recommend_movies_cosine_User
from website.keras import recommend_movies_keras

from website.search import search_title, clean_title
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import sqlite3
from tqdm import tqdm
from sklearn.metrics import average_precision_score,mean_squared_error, mean_absolute_error
conn = sqlite3.connect(r'C:\Users\ttzur\Desktop\WebWithFlask\instance\database.db')
sql_query = pd.read_sql_query ('''
                               SELECT
                               *
                               FROM ratings
                               ''', conn)

ratings = pd.DataFrame(sql_query, columns = ['id','user_id','movie_id', 'rating'])



def recommend_movies_hybrid_User(user_id, knn_weight=0.4, keras_weight=0.6, num_recommendations=50):
    _, knn_scores,knn_results = recommend_movies_knn_User(user_id)
    
    _, keras_scores, keras_results = recommend_movies_keras(user_id)

    knn_results = knn_results.copy()
    
    keras_results = keras_results.copy()

    # Normalize the similarity scores
    knn_scores = knn_scores / np.max(knn_scores)*1
    
    keras_scores = keras_scores / np.max(keras_scores)*1
    
    # Combine the results and similarity scores
    knn_results["similarity_score"] = [score * knn_weight for score in knn_scores]
    
    keras_results["similarity_score"] = [score * keras_weight for score in keras_scores]

    # Merge the results based on 'movie_id'
    
    merged_results = knn_results.merge(keras_results, on=['movie_id', 'title', 'genre', 'imdb_id', 'tmdb_id'], how='outer', suffixes=('_knn','_keras'))
    merged_results["title"] = merged_results["title"].apply(clean_title)
    # Fill missing values with 0
    merged_results['similarity_score_knn'] = merged_results['similarity_score_knn'].fillna(0)
    merged_results['similarity_score_keras'] = merged_results['similarity_score_keras'].fillna(0)
    
    # Add a new column that is the sum of the similarity_score_knn and similarity_score_keras columns
    merged_results['similarity_score'] = merged_results['similarity_score_knn'] + merged_results['similarity_score_keras']
    merged_results = merged_results.groupby(['movie_id', 'title', 'genre', 'imdb_id', 'tmdb_id'], as_index=False).agg({'similarity_score': 'sum'})
   
    
    # Sort by similarity score and select the top num_recommendations
    final_results = merged_results.sort_values(by='similarity_score', ascending=False).head(num_recommendations)
    
    # Drop unnecessary column
    final_results = final_results.drop(columns=['similarity_score'])
    final_results['movie_id'] = final_results['movie_id'].astype(int)
    final_results['title'] = final_results['title'].str.title()
    final_results = final_results.head(10)
    return final_results.reset_index(drop=True)


# # Step 1: Split the ratings dataset
# train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

# # Step 2: Generate recommendations for users in the test dataset
# test_users = test_ratings['user_id'].unique()
# recommendations = {}
# for user_id in tqdm(test_users, desc="Generating recommendations"):
#     rec_movies = recommend_movies_hybrid_User(user_id)
#     recommendations[user_id] = set(rec_movies['movie_id'])

# # Step 3: Compare the recommendations with the actual movie ratings in the test dataset
# true_positives = {}
# for user_id, recommended_movies in recommendations.items():
#     actual_movies = set(test_ratings[test_ratings['user_id'] == user_id]['movie_id'])
#     true_positives[user_id] = recommended_movies.intersection(actual_movies)

# # Step 4: Calculate precision, recall, and F1 score
# precisions = [len(tp) / len(recommendations[user_id]) for user_id, tp in true_positives.items()]
# recalls = [len(tp) / len(set(test_ratings[test_ratings['user_id'] == user_id]['movie_id'])) for user_id, tp in true_positives.items()]

# precision = np.mean(precisions)
# recall = np.mean(recalls)
# f1_score = 2 * (precision * recall) / (precision + recall)




# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1_score)

# user_id = 2
# start_time = time.time()
# recommend_movies_hybrid_User(user_id)
# hybrid_time = time.time() - start_time
# print("Hybrid Time: {:.2f} seconds".format(hybrid_time))





