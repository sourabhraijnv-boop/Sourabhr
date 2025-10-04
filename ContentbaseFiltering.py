import numpy as np
import numpy.ma as ma
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_KERAS'] = '1'
import pandas as pd
import tensorflow as tf
from tensorflow import keras 
from keras.models import Sequential
from keras import layers, ops, Model
from keras.activations import relu
from keras.layers import Dense, Input, Layer, Dot
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
pd.set_option("display.precision", 1)
pd.set_option('future.no_silent_downcasting', True)

movies = pd.read_csv("C:/Users/soura/ml-latest-small/ml-latest-small/movies.csv")
ratings = pd.read_csv("C:/Users/soura/ml-latest-small/ml-latest-small/ratings.csv")
tags = pd.read_csv("C:/Users/soura/ml-latest-small/ml-latest-small/tags.csv")
links = pd.read_csv("C:/Users/soura/ml-latest-small/ml-latest-small/links.csv")
#print(movies.head())
def load_data():
    avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
    ratings_array = avg_ratings.to_numpy(dtype=object)
    ratings_dataframe= pd.DataFrame(ratings_array, columns=['movieId', 'ave rating'])

    #print(ratings_dataframe.head())
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(object)
    movie_years = movies[["movieId", "year"]].to_numpy(dtype=object)
    movie_years_dataframe = pd.DataFrame(movie_years, columns=['movieId', 'years'])

    genre_dummies = movies["genres"].str.get_dummies(sep='|')
    movies_exp = pd.concat([movies, genre_dummies], axis=1)
    movies_expanded = movies_exp
    movies_expanded = movies_expanded.drop(columns=movies_expanded.columns[[1, 2, 3, 4]])
    #print(movies_expanded[:5])

    movie_data = pd.merge(movie_years_dataframe,ratings_dataframe,on='movieId', how='inner')
    movie_data = pd.merge(movie_data,movies_expanded,on='movieId', how='inner')
    movie_data = ratings[['movieId']].merge(movie_data, on='movieId', how='left')
    df = movie_data.fillna(2000)
    movie_data = df
    
    movie_arr = movie_data.to_numpy(dtype=float) #movie_train set

    # Merge ratings with expanded genres
    ratings_merged = ratings.merge(movies_exp, on="movieId")

    # Multiply rating by each genre column (so ratings are weighted into genres)
    for genre in genre_dummies.columns:
        ratings_merged[genre] = ratings_merged[genre] * ratings_merged["rating"]

    user_summary = ratings_merged.groupby("userId").agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean"),
        **{genre: (genre, "mean") for genre in genre_dummies.columns}
    ).reset_index()

    user_summary = ratings[["userId"]].merge(user_summary, on="userId", how="left")
    user_summary = user_summary.drop(columns=user_summary.columns[[3]])
    
    user_arr = user_summary.to_numpy(dtype=float)
    y = ratings['rating'].to_numpy(dtype=float)
    return movie_arr, user_arr, y, movie_data, user_summary

def scale_data(movie_arr, user_arr, y):
    num_user_features = user_arr.shape[1] - 3
    num_movie_features = movie_arr.shape[1] - 1

    scalermovie = StandardScaler()
    scalermovie.fit(movie_arr)
    movie_arr = scalermovie.transform(movie_arr)
    
    scaleuser = StandardScaler()
    scaleuser.fit(user_arr)
    user_arr = scaleuser.transform(user_arr)

    scaletarg = MinMaxScaler((-1,1))
    scaletarg.fit(y.reshape(-1,1))
    y = scaletarg.transform(y.reshape(-1,1))
    return movie_arr, user_arr, y, num_movie_features, num_user_features

def create_model(num_movie_features, num_user_features):
    num_outputs = 32
    tf.random.set_seed(1)
    user_NN = Sequential([
        Dense(256, activation = 'relu'),
        Dense(128, activation = 'relu'),
        Dense(num_outputs)
    ])

    movie_NN = Sequential([
        Dense(256, activation = 'relu'),
        Dense(128, activation = 'relu'),
        Dense(num_outputs)
    ])

    class L2NormalizeLayer(Layer):
        def call(self, inputs):
            return tf.math.l2_normalize(inputs)


    input_user = Input(shape=[num_user_features])
    vu = user_NN(input_user)
    vu = L2NormalizeLayer()(vu)

    input_movie = Input(shape=[num_movie_features])
    vm = movie_NN(input_movie)
    vm = L2NormalizeLayer()(vm)

    output = Dot(axes=1)([vu, vm])
    model = Model([input_user, input_movie], output)

    tf.random.set_seed(1)
    cost_fn = MeanSquaredError()
    opt = Adam(learning_rate = 0.01, clipnorm=0.5)
    model.compile(optimizer= opt, loss= cost_fn, metrics=['accuracy'])

    model_m = Model(input_movie, vm)
    
    #model_m.summary()

    return model, model_m

movie_arr, user_arr, y, movie_data, user_summary = load_data()
movie_arr, user_arr, y, num_movie_features, num_user_features = scale_data(movie_arr, user_arr, y)
movie_train, movie_test = train_test_split(movie_arr, train_size=0.50, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_arr, train_size=0.50, shuffle=True, random_state=1)
y_train, y_test = train_test_split(y, train_size=0.50, shuffle=True, random_state=1)

model, model_m = create_model(num_movie_features, num_user_features)
scaleMovie =StandardScaler()
scaleMovie.fit(movie_arr)
smovie_vecs = scaleMovie.transform(movie_arr)
vms = model_m.predict(smovie_vecs[:,1:])
print(f"size of all predicted movie feature vectors: {vms.shape}")


tf.random.set_seed(1)
model.fit([user_train[:, 3:], movie_train[:, 1:]], y_train, epochs = 1)
loss, accuracy = model.evaluate([user_test[:, 3:], movie_test[:, 1:]], y_test)
print(f"Loss : {loss}\nAccuracy : {accuracy}") 

#prediction for new user
def Predict_newuser(user_vec, model, y, movie_arr):
    user_vec = np.repeat(user_vec,len(movie_arr), axis=0)

    scalerUser = StandardScaler()
    scalerUser.fit(user_vec)
    suser_vecs = scalerUser.transform(user_vec)

    scaleMovie =StandardScaler()
    scaleMovie.fit(movie_arr)
    smovie_vecs = scaleMovie.transform(movie_arr)

    y_p = model.predict([suser_vecs[:, 3:], smovie_vecs[:, 1:]])
    scaleTarget = StandardScaler()
    scaleTarget.fit(y)
    y_pu = scaleTarget.inverse_transform(y_p)

    sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()
    sorted_ypu = y_pu[sorted_index]
    sorted_movie = movie_arr[sorted_index]

    print(sorted_ypu[:10, :], sorted_movie[:10, :])
    return y_pu

#user_vec = np.array([[5000, 3, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
#y_pu = Predict_newuser(user_vec, model,y , movie_arr)
def predict_ratings(uid, model, y, movie_arr):
    movie_ar, user_ar, y_a, movie_data, user_summary = load_data()
    indices = np.where(user_ar[:, 0] == uid)[0]
    user_vec = user_ar[indices[0]].reshape(1,-1)
    user_vec = np.repeat(user_vec,len(movie_arr), axis=0)
    y_vecs = np.zeros(len(movie_arr))
    np.put(y_vecs, indices, y_a)
    scalerUser = StandardScaler()
    scalerUser.fit(user_vec)
    suser_vecs = scalerUser.transform(user_vec)

    scaleMovie =StandardScaler()
    scaleMovie.fit(movie_arr)
    smovie_vecs = scaleMovie.transform(movie_arr)

    y_p = model.predict([suser_vecs[:, 3:], smovie_vecs[:, 1:]])
    scaleTarget = StandardScaler()
    scaleTarget.fit(y)
    y_pu = scaleTarget.inverse_transform(y_p)   

    sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()
    sorted_ypu = y_pu[sorted_index]
    sorted_movie = movie_ar[sorted_index]
    sorted_user = user_vec[sorted_index]
    sorted_y = y_vecs[sorted_index]

    print(sorted_ypu[:10, :], sorted_movie[:10, :], sorted_user[:10, :], sorted_y[:10])   

#predict_ratings(2, model, y, movie_arr)

def sq_dist(a,b):
    return sum(np.square(a-b))
#find closest movies
count = 50
dim = 1000
dist = np.zeros((dim,dim))

for i in range(dim):
    for j in range(dim):
        dist[i,j] = sq_dist(vms[i, :], vms[j, :])

m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0])) #mask digonal
disp = [["movie1", "genres", "movie2", "genres"]]

for i in range(count):
    min_idx = np.argmin(m_dist[i])
    movie1_id = int(movie_arr[i,0])
    movie2_id = int(movie_arr[min_idx,0])
    print(movie1_id,movie2_id)
    disp.append( [movies.loc[movie_data["movieId"] == movie1_id+1, 'title'].iloc[0], movies.loc[movie_data["movieId"] == movie1_id+1, 'genres'].iloc[0],
                  movies.loc[movie_data["movieId"] == movie2_id+1, 'title'].iloc[0], movies.loc[movie_data["movieId"] == movie2_id+1, 'genres'].iloc[0]])
    
table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
print(table)