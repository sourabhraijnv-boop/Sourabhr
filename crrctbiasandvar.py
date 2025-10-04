import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

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

model = LinearRegression()

reg_params = [10, 5, 2, 1, 0.5, 0.2, 0.1]