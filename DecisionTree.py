import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_train = np.array([[1,1,1],[0,0,1],[0,1,0],[1,0,1],[1,1,1],[1,1,0],[0,0,0],[1,1,0],[0,1,0],[0,1,0]])
y_train = np.array([1,1,0,0,1,1,0,1,0,0])

def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1-p)*np.log2(1-p)

def split_indices(X, index_feature):
    left_ind = []
    right_ind = []
    for i,x in enumerate(X):
        if x[index_feature] == 1:
            left_ind.append(i)
        else:
            right_ind.append(i)
    return left_ind, right_ind

def weight_entropy(X,y,left_ind,right_ind):
    w_left = len(left_ind)/len(X)
    w_right = len(right_ind)/len(X)
    p_left = sum(y[left_ind])/len(left_ind)
    p_right = sum(y[right_ind])/len(right_ind)

    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy

def information_gain(X,y,left_ind,right_ind):
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)
    w_entropy = weight_entropy(X,y,left_ind,right_ind)
    return h_node - w_entropy

for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
    left_ind, rght_ind = split_indices(x_train,i)
    i_gain = information_gain(x_train,y_train,left_ind,rght_ind)
    print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")
 
        