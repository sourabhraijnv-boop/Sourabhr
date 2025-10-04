import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_KERAS'] = '1'
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def cofi_cost_func(X, W, b, Y, R, lambda_):
    nm, nu = Y.shape
    J = 0
    for j in range(nu):
        w = W[j,:]
        b_j = b[0,j]
        for i in range(nm):
            x = X[i,:]
            y = Y[i,j]
            r = R[i,j]
            J += np.square(r * (np.dot(w,x) + b_j - y ) )
    
    J = J/2
    J += (lambda_/2) * (np.sum(np.square(W)) + np.sum(np.square(X)))

    return J

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

df = pd.read_csv("C:/Users/soura/small_movie_list.csv")
movieList = df["title"].values

num_movies1 = df.shape[0]
my_ratings = np.zeros(num_movies1)

my_ratings[2700] = 5
my_ratings[929] = 5
my_ratings[246] = 5
my_ratings[2716] = 3
my_ratings[1150] = 5
my_ratings[382] = 2
my_ratings[366] = 5
my_ratings[622] = 5
my_ratings[988] = 3
my_ratings[2925] = 1
my_ratings[2937] = 1
my_ratings[793] = 5
my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]
'''
print('\nNew user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0 :
        print(f'Rated {my_ratings[i]} for {df.loc[i, "title"]}')
'''
Y = np.random.randint(0, 6, size=(4778, 443))  
R = np.random.randint(0, 2, size=(4778, 443))
Y = np.c_[my_ratings, Y]
R = np.c_[(my_ratings != 0).astype(int), R]
row_means = Y.mean(axis=1, keepdims=True)
Y_norm = Y - row_means

num_movies, num_users = Y.shape
num_feature = 100
tf.random.set_seed(1234)
W = tf.Variable(tf.random.normal((num_users, num_feature), dtype=tf.float64), name='W')
X = tf.Variable(tf.random.normal((num_movies, num_feature), dtype=tf.float64), name='X')
b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')
optimizer = keras.optimizers.Adam(learning_rate=1e-1)
iterations = 200
lambda_ = 1
# usig gradient descent using tensorflow with adam optimizer
'''
for iter in range(iterations):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func_v(X,W,b, Y_norm,R, lambda_)

    grads = tape.gradient(cost_value, [X,W,b])

    optimizer.apply_gradients(zip(grads, [X,W,b])) 

    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")
'''
row_means = np.mean(Y_norm, axis=1, keepdims=True)

# Broadcast to same shape as A
Ymean = np.tile(row_means, (1, Y_norm.shape[1]))

#making predictions
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
#restoring mean
pm = p + Ymean

my_prediction = pm[:,0]
#sort prediction
ix = tf.argsort(my_prediction, direction='DESCENDING')
'''
for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'Predicting rating {my_prediction[j]:0.2f} for movie {movieList[j]}')

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Original {my_ratings[i]}, Predicted {my_prediction[i]:0.2f} for {movieList[i]}')
'''
print(df.head())
filter = (df["number of ratings"] > 20)
df["pred"] = my_prediction
df = df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False)
print(df.head())