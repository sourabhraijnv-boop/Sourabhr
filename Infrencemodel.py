import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense
import logging 
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X= np.array([[0.7,0.5],[0.8, 0.8],[1.9,2.0],[2,1.9],[1.5,1.3],[1.4,1.6],[1.5,1.5]])
Y= np.array([[0],[0],[0],[0],[1],[1],[1]])
##### normalization of data
#norm_1=tf.keras.layers.Normalization(axis=-1)
#norm_1=adapt(X)
#Xn=norm_1(X)

#### normalization end
tf.random.set_seed(1234)
model = Sequential(
    [
        Dense(3,activation='sigmoid', name = 'L1'),
        Dense(1,activation='sigmoid', name = 'L2')
    ]
)

model.compile(
    loss= tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(X,Y,epochs=10)
w1,b1=model.get_layer('L1').get_weights()
w2,b2=model.get_layer('L2').get_weights()

print(f"w1 = {w1}\t", f"b1 = {b1}\n")
print(f"w2 = {w2}\t", f"b2 = {b2}\n")