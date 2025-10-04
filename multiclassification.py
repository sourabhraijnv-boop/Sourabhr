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

centers = [[-5, 2 ], [-2, -2], [1, 2], [5, -2]]
x_train, y_tarin = make_blobs(n_samples=100, centers=centers, cluster_std=1.0, random_state=30)

tf.random.set_seed(1234)
model = Sequential(
    [
        Dense(2, activation='relu', name = "L1"),
        Dense(4, activation='linear', name = "L2")
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)
model.fit(x_train,y_tarin,epochs=200)

l1=model.get_layer("L1")
w1,b1 = l1.get_weights()
l2=model.get_layer("L2")
w2,b2 = l2.get_weights()
print(w1,b1,w2,b2)