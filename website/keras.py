import numpy as np
import pandas as pd
import sqlite3
import re
from website.search import search_title
import tensorflow as tf
import time
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dot, Flatten, Concatenate, Dense
from keras.optimizers import Adam
from collections import defaultdict
from tqdm import tqdm
from keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing

conn = sqlite3.connect(r'C:\Users\ttzur\Desktop\WebWithFlask\instance\database.db')# get all movies from the database
sql_query = pd.read_sql_query('''
                               SELECT
                               *
                               FROM movie
                               ''', conn)

movies = pd.DataFrame(sql_query, columns=['id', 'title','genre'])

movies = movies.rename(columns={'id': 'movie_id'})

sql_query = pd.read_sql_query('''
                               SELECT
                               *
                               FROM ratings
                               ''', conn)

ratings = pd.DataFrame(
    sql_query, columns=['id', 'user_id', 'movie_id', 'rating','timestamp'])

sql_query = pd.read_sql_query ('''
                               SELECT
                               *
                               FROM link
                               ''', conn)
links = pd.DataFrame(sql_query, columns = ['id','movie_id','imdb_id', 'tmdb_id'])
links['imdb_id'] = links['imdb_id'].fillna('')
links['tmdb_id'] = links['tmdb_id'].fillna('')

sql_query = pd.read_sql_query('''
                               SELECT
                               *
                               FROM tag
                               ''', conn)


tags = pd.DataFrame(sql_query, columns=['movie_id', 'tag'])


user_ids = ratings["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

movie_ids = ratings['movie_id'].unique().tolist()

movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

ratings["user"] = ratings["user_id"].map(user2user_encoded)
ratings["movie"] = ratings["movie_id"].map(movie2movie_encoded)

num_ratings = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
ratings["rating"] = ratings["rating"].values.astype(np.float32)
# min and max ratings will be used to normalize the ratings later
min_rating = min(ratings["rating"])
max_rating = max(ratings["rating"])



ratings = ratings.sample(frac=1, random_state=42)
x = ratings[["user", "movie"]].values
# Normalize the targets between 0 and 1. Makes it easy to train.
y = ratings["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# Assuming training on 90% of the data and validating on 10%.
train_indices = int(0.9 * ratings.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

EMBEDDING_SIZE = 50

class RecommenderNet(keras.Model):
    def __init__(self, num_ratings, num_movies, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_ratings = num_ratings
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_ratings,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_ratings, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)


model = RecommenderNet(num_ratings, num_movies, EMBEDDING_SIZE)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)

# history = model.fit(
#      x=x_train,
#      y=y_train,
#      batch_size=64,
#      epochs=6,
#      verbose=1,
#      validation_data=(x_val, y_val),
# )

# model.save("model", save_format="tf")

model=load_model(r'C:\Users\ttzur\Desktop\WebWithFlask\model')

def recommend_movies_keras(user_id):
    movies_watched_by_user = ratings[ratings.user_id == user_id]
    all_movie_ids = list(set(movie2movie_encoded.keys()))
    all_movie_ids.remove(max(all_movie_ids))
    all_movie_ids_encoded = [[movie2movie_encoded.get(x)] for x in all_movie_ids]
    

    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(all_movie_ids_encoded), all_movie_ids_encoded)
    ).astype(np.int64)

    predicted_ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = predicted_ratings.argsort()[-50:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(all_movie_ids_encoded[x][0]) for x in top_ratings_indices
    ]

    recommended_movies = movies[movies["movie_id"].isin(recommended_movie_ids)]
    top_sim_scores = predicted_ratings[top_ratings_indices]
    top_sim_scores = pd.Series(top_sim_scores, index=recommended_movie_ids)
    recommended_movies = recommended_movies.merge(links, on='movie_id', how='left')[["movie_id", "title", "genre", "imdb_id", "tmdb_id"]]
    recommended_movies_hybrid = recommended_movies
    recommended_movies = recommended_movies.head(10)
    return recommended_movies, top_sim_scores, recommended_movies_hybrid

# # Step 1: Make predictions on the validation data
# y_pred = model.predict(x_val)

# # Step 2: Denormalize the predicted ratings
# y_pred_denorm = y_pred * (max_rating - min_rating) + min_rating

# # Step 3: Get the true ratings and denormalize them
# y_val_denorm = y_val * (max_rating - min_rating) + min_rating

# # Step 4: Compute the squared differences
# squared_diff = np.square(y_val_denorm - y_pred_denorm)

# # Step 5: Calculate the mean of these squared differences
# mean_squared_diff = np.mean(squared_diff)

# # Step 6: Take the square root to get the RMSE
# rmse = np.sqrt(mean_squared_diff)

# print("RMSE:", rmse)

# user_id=2
# start_time = time.time()
# recommend_movies_keras(user_id)
# keras_time = time.time() - start_time
# print("Keras Time: {:.2f} seconds".format(keras_time))




# def plot_learning_curves(history):
#     plt.figure(figsize=(10, 5))
    
#     # Plot training & validation loss values
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper right')
    
    # If your model has a metric like accuracy, you can plot it as well
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')
    
#     plt.tight_layout()
#     plt.show()

# plot_learning_curves(history)