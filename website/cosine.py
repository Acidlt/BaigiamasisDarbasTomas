import sqlite3
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from website.search import search_title

# Load your movies data, which should include genres and/or tags
conn = sqlite3.connect(r'C:\Users\ttzur\Desktop\WebWithFlask\instance\database.db')
sql_query = pd.read_sql_query('''
                               SELECT
                               *
                               FROM movie
                               ''', conn)

movies = pd.DataFrame(sql_query, columns=['id', 'title','genre'])

sql_query = pd.read_sql_query('''
                               SELECT
                               *
                               FROM tag
                               ''', conn)


tags = pd.DataFrame(sql_query, columns=['movie_id', 'tag'])

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

# Group tags by movie_id and join them into a single string
grouped_tags = tags.groupby('movie_id')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# Merge the movie and tag DataFrames on the 'movie_id' column
movies = movies.merge(grouped_tags, left_on='id', right_on='movie_id', how='left').drop('movie_id', axis=1)

# Fill any NaN values in the 'tag' column with an empty string
movies['tag'] = movies['tag'].fillna('')

# Combine genres and tags into a single 'features' column
movies['features'] = movies['genre'] + ' ' + movies['tag']

# Compute the TF-IDF matrix for the feature column
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies['features'])

# Compute the cosine similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movies_cosine(title, num_recommendations=10):
    
    movieTitleSearch=search_title(title)
    movie_id=movieTitleSearch.iloc[0]["movie_id"]

    movie_index = movies[movies['id'] == movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    recommended_movie_indices = [i[0] for i in sim_scores]
    top_sim_scores = [x[1] for x in sim_scores]
    result = movies.iloc[recommended_movie_indices][['id', 'title','genre']]
    result = result.merge(links, on='id',how='left')[["id", "title","genre", "imdb_id", "tmdb_id"]]
    result["title"] = result["title"].str.title()
    return result,movieTitleSearch,top_sim_scores

def recommend_movies_cosine_User(user_id, num_recommendations=10):
    # Compute the average similarity scores for all movies based on the user's rated movies
    avg_sim_scores = [(i, 0) for i in range(len(movies))]  # initialize with movie indices
    user_ratings = ratings[ratings['user_id'] == user_id]

    for idx, row in user_ratings.iterrows():
        movie_id = row['movie_id']
        rating = row['rating']
        movie_indices = movies[movies['id'] == movie_id].index
        if not movie_indices.empty:
            movie_index = movie_indices[0]
            sim_scores = list(enumerate(cosine_sim[movie_index]))

            # Adjust similarity scores based on user ratings
            for i, score in sim_scores:
                avg_sim_scores[i] = (i, avg_sim_scores[i][1] + score * rating)

    # Normalize the average similarity scores
    avg_sim_scores = [(i, score / len(user_ratings)) for i, score in avg_sim_scores]

    # Sort the average similarity scores
    avg_sim_scores = sorted(avg_sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top recommendations
    recommended_movie_indices = [i for i, score in avg_sim_scores[:num_recommendations]]
    recommended_sim_scores = [score for i, score in avg_sim_scores[:num_recommendations]]

    result = movies.iloc[recommended_movie_indices][['id', 'title', 'genre']]
    result = result.merge(links, on='id', how='left')[["id", "title", "genre", "imdb_id", "tmdb_id"]]
    result = result.rename(columns={"id": "movie_id"})
    result["title"] = result["title"].str.title()

    return result, recommended_sim_scores

# user_id=2
# start_time = time.time()
# recommend_movies_cosine_User(user_id)
# cosine_time = time.time() - start_time
# print("Cosine Time: {:.2f} seconds".format(cosine_time))


# # Step 1: Split the ratings dataset
# train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

# # Step 2: Generate recommendations for users in the test dataset
# test_users = test_ratings['user_id'].unique()
# recommendations = {}
# for user_id in tqdm(test_users, desc="Generating recommendations"):
#     rec_movies, _ = recommend_movies_cosine_User(user_id)
#     recommendations[user_id] = set(rec_movies['movie_id'])

# # Step 3: Compare the recommendations with the actual movie ratings in the test dataset
# true_positives = {}
# total_predicted = 0
# total_actual = 0
# for user_id, recommended_movies in recommendations.items():
#     actual_movies = set(test_ratings[test_ratings['user_id'] == user_id]['movie_id'])
#     true_positives[user_id] = recommended_movies.intersection(actual_movies)
#     total_predicted += len(recommended_movies)
#     total_actual += len(actual_movies)

# # Step 4: Calculate precision, recall, and F1 score
# precision = np.mean([len(tp) / len(recommendations[user_id]) for user_id, tp in true_positives.items()])
# recall = np.mean([len(tp) / len(set(test_ratings[test_ratings['user_id'] == user_id]['movie_id'])) for user_id, tp in true_positives.items()])
# f1_score = 2 * (precision * recall) / (precision + recall)

# # Step 5: Calculate accuracy
# tp_sum = sum([len(tp) for tp in true_positives.values()])
# accuracy = tp_sum / total_actual

# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1_score)
# print("Accuracy:", accuracy)