import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_KERAS'] = '1'
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input
from keras import Sequential
from keras.activations import sigmoid
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

x_train = np.array([[1.0],[2.0]],dtype=np.float32)
y_train = np.array([[300.0],[500.0]],dtype=np.float32)
linear_layer= tf.keras.layers.Dense(units=1,activation = 'linear',)
linear_layer.get_weights()
a1=linear_layer(x_train[0].reshape(1,1))
print(a1)
w, b= linear_layer.get_weights()
print(f"w = {w}, b={b}")
set_w = np.array([[200]])
set_b =np.array([100])
linear_layer.set_weights([set_w,set_b])
print(linear_layer.get_weights())
a1=linear_layer(x_train[0].reshape(1,1))
print(a1)
alin = np.dot(set_w,x_train[0].reshape(1,1)) + set_b
print(alin)
X_train = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)

model = Sequential(
    [
        tf.keras.layers.Dense(1, input_dim=1, activation = 'sigmoid', name= 'L1')
    ]
)
model.summary()
logistic_layer=model.get_layer('L1')
w,b = logistic_layer.get_weights()
print(w,b)
w1=np.array([[2]])
b1=np.array([-4.5])
logistic_layer.set_weights([w1,b1])
a1 = model.predict(X_train[0].reshape(1,1))
print(a1)
alog = np.dot(w1,X_train[0].reshape(1,1)) + b1
alog = 1/(1+np.exp(-alog))
print(alog)