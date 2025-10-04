import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.activations import linear, relu, sigmoid, softmax
from matplotlib.widgets import Slider
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import logging 
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def my_softmax(z):
    ez = np.exp(z)
    sm = ez/np.sum(ez)
    return(sm)

plt.close("all")
centers = [[-5, 2 ], [-2, -2], [1, 2], [5, -2]]
x_train, y_tarin = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0, random_state=30)
model = Sequential(
    [
        Dense(25, activation='relu'),
        Dense(15, activation='relu'),
        Dense(4, activation='softmax')
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    x_train,y_tarin, epochs=10
)

p_nonpreferd = model.predict(x_train)
print(p_nonpreferd [:2])
print("largest value", np.max(p_nonpreferd), "smallest value", np.min(p_nonpreferd))

sm_preferd = tf.nn.softmax(p_nonpreferd).numpy()
print(f"two example output vectors:\n {sm_preferd[:2]}")
print("largest value", np.max(sm_preferd), "smallest value", np.min(sm_preferd))

for i in range(5):
    print( f"{p_nonpreferd[i]}, category: {np.argmax(p_nonpreferd[i])}")

