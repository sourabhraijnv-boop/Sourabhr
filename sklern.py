import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
np.set_printoptions(precision=2)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

data = np.loadtxt('data12.txt', delimiter=',')
x= data[:,0]
y= data[:,1]
# convert 1d array to 2d
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

#split data in 60:20:20 for train cross validation and test
x_train, x_, y_tarin, y_ = train_test_split(x,y,test_size=0.40,random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_,y_,test_size=0.50,random_state=1)

del x_,y_
'''
print(f"shape of training set x : {x_train.shape}")
print(f"shape of training set y : {y_tarin.shape}")
print(f"shape of cv set x : {x_cv.shape}")
print(f"shape of cv set y : {y_cv.shape}")
print(f"shape of test set x : {x_test.shape}")
print(f"shape of test set y : {y_test.shape}")
'''

#feature scaling
scaler_linear = StandardScaler()
x_train_scaled = scaler_linear.fit_transform(x_train)

'''
print(f"mean of training set: {scaler_linear.mean_.squeeze():.2f}")
print(f"standerd deviation of train set: {scaler_linear.scale_.squeeze():.2f}")
'''
#train model

linear_model = LinearRegression()
linear_model.fit(x_train_scaled,y_tarin)
yhat = linear_model.predict(x_train_scaled)
'''
print(f"MSE on train data: {mean_squared_error(y_tarin,yhat)/2}")

total_mse = 0
for i in range(len(yhat)):
    sq_err = (yhat[i] - y_tarin[i])**2
    total_mse += sq_err
mse = total_mse /(2*len(yhat))

print(f"MSE using loop : {mse.squeeze()}")
'''
x_cv_scaled = scaler_linear.transform(x_cv)
'''
print(f"mean of training set: {scaler_linear.mean_.squeeze():.2f}")
print(f"standerd deviation of train set: {scaler_linear.scale_.squeeze():.2f}")
'''
yhatcv = linear_model.predict(x_cv_scaled)

#print(f"MSE on cv data: {mean_squared_error(y_cv,yhatcv)/2}")

# adding polynomial feature to reduce bias

poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_mapped = poly.fit_transform(x_train)
#print(x_train_mapped[:5])

scaler_poly = StandardScaler()
x_train_mapped_scaled = scaler_poly.fit_transform(x_train_mapped)

#print(x_train_mapped_scaled[:5])

model = LinearRegression()
model.fit(x_train_mapped_scaled,y_tarin)
yhat = model.predict(x_train_mapped_scaled)
#print(f"MSE using poly: {mean_squared_error(y_tarin,yhat)/2}")
x_cv_mapped =poly.transform(x_cv)
x_cv_mapped_scaled = scaler_poly.transform(x_cv_mapped)
yhatcv = model.predict(x_cv_mapped_scaled)
#print(f"MSE on cv data using poly: {mean_squared_error(y_cv,yhatcv)/2}")

#using higher degrees to see mse

train_mses = []
cv_mses = []
models = []
polys = []
scalers = []

for degree in range(1,11):

    poly = PolynomialFeatures(degree, include_bias=False)
    x_train_mapped = poly.fit_transform(x_train)
    polys.append(poly)

    scaler_poly = StandardScaler()
    x_train_mapped_scaled = scaler_poly.fit_transform(x_train_mapped)
    scalers.append(scaler_poly)

    model = LinearRegression()
    model.fit(x_train_mapped_scaled,y_tarin)
    models.append(model)

    yhat = model.predict(x_train_mapped_scaled)
    train_mse = mean_squared_error(y_tarin,yhat)/2
    train_mses.append(train_mse)

    x_cv_mapped = poly.transform(x_cv)
    x_cv_mapped_scaled = scaler_poly.transform(x_cv_mapped)

    yhatcv = model.predict(x_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv,yhatcv)/2
    cv_mses.append(cv_mse)

'''
print(f"train mses : {train_mses}\n")
print(f"cv mses : {cv_mses}")
'''
# select degree wrt lowest cv_mse
degree = np.argmin(cv_mses)+1
print(f"Lowest cv mse found in the model with degeree={degree}")

#using this model on test set
x_test_mapped = polys[degree-1].transform(x_test)
x_test_mapped_scaled = scalers[degree-1].transform(x_test_mapped)
yhattest = models[degree-1].predict(x_test_mapped_scaled)
testmse = mean_squared_error(yhattest,y_test)/2

print(f"train mse: {train_mses[degree-1]:.2f}")
print(f"cv mse: {cv_mses[degree-1]:.2f}")
print(f"test mse: {testmse:.2f}")