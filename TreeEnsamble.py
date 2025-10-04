import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
RANDOM_STATE = 55
df = pd.read_csv("heart.csv")

cat_variables = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

#one hot encoding of categorial variables
df = pd.get_dummies(data=df, prefix= cat_variables, columns=cat_variables)

#taking feature only with heartdisease
features = [x for x in df.columns if x not in 'HeartDisease']

x_train, x_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size=0.8, random_state=RANDOM_STATE)

'''
print(f'train sample: {len(x_train)}')
print(f'validation sample: {len(x_val)}')
print(f'target proportion: {sum(y_train)/len(y_train):.4f}')
'''
#1 Decision tree

min_sample_split_list = [2, 10, 30, 50, 100, 200, 300, 700]
max_depth_list = [1, 2, 3, 4, 8, 16, 32, 64, None]

# calculate min sample split using graph
'''
accuracy_list_train = []
accuracy_list_val = []
for min_sample_split in min_sample_split_list:
    model = DecisionTreeClassifier(min_samples_split= min_sample_split, random_state=RANDOM_STATE).fit(x_train,y_train)
    prediction_train = model.predict(x_train)
    prediction_val = model.predict(x_val)
    accuracy_train = accuracy_score(prediction_train,y_train)
    accuracy_val = accuracy_score(prediction_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrices')
plt.xlabel('min_sample_split')
plt.ylabel('accuracy')
plt.xticks(ticks= range(len(min_sample_split_list)), labels=min_sample_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train', 'Validation'])
plt.show()
'''
#calculate max depth using graph
'''
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE).fit(x_train,y_train)
    prediction_train = model.predict(x_train)
    prediction_val = model.predict(x_val)
    accuracy_train = accuracy_score(prediction_train,y_train)
    accuracy_val = accuracy_score(prediction_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrices')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks= range(len(max_depth_list)), labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train', 'Validation'])
plt.show()
'''
decision_tree_model = DecisionTreeClassifier(min_samples_split=50, max_depth=3, random_state=RANDOM_STATE).fit(x_train,y_train)
'''
print(f"Metrices train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(x_train),y_train):.4f}")
print(f"Metrices validation:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(x_val),y_val):.4f}")
'''
#Random forest
n_estimator_list = [10, 50, 100, 500]

# calculate max depth using random forest
'''
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    model = RandomForestClassifier(max_depth=max_depth, random_state=RANDOM_STATE).fit(x_train,y_train)
    prediction_train = model.predict(x_train)
    prediction_val = model.predict(x_val)
    accuracy_train = accuracy_score(prediction_train,y_train)
    accuracy_val = accuracy_score(prediction_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrices')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks= range(len(max_depth_list)), labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train', 'Validation'])
plt.show()
'''
#calculate min split using random forest
'''
accuracy_list_train = []
accuracy_list_val = []
for min_sample_split in min_sample_split_list:
    model = RandomForestClassifier(min_samples_split= min_sample_split, random_state=RANDOM_STATE).fit(x_train,y_train)
    prediction_train = model.predict(x_train)
    prediction_val = model.predict(x_val)
    accuracy_train = accuracy_score(prediction_train,y_train)
    accuracy_val = accuracy_score(prediction_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrices')
plt.xlabel('min_sample_split')
plt.ylabel('accuracy')
plt.xticks(ticks= range(len(min_sample_split_list)), labels=min_sample_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train', 'Validation'])
plt.show()
'''
#calulate estimators using random forest
'''
accuracy_list_train = []
accuracy_list_val = []
for n_estiamators in n_estimator_list:
    model = RandomForestClassifier(n_estimators=n_estiamators, random_state=RANDOM_STATE).fit(x_train,y_train)
    prediction_train = model.predict(x_train)
    prediction_val = model.predict(x_val)
    accuracy_train = accuracy_score(prediction_train,y_train)
    accuracy_val = accuracy_score(prediction_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrices')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xticks(ticks= range(len(n_estimator_list)), labels=n_estimator_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train', 'Validation'])
plt.show()
'''
random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_split=10).fit(x_train,y_train)
#print(f"Metrices train:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(x_train),y_train):.4f}\nMetrices test:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(x_val),y_val):.4f}" )

#Xgboost

n = int(len(x_train)*0.8)
x_train_fit, x_train_eval, y_train_fit, y_train_eval = x_train[:n], x_train[n:], y_train[:n], y_train[n:]
xgb_model = XGBClassifier(n_estimators = 500, learning_rate = 0.1, verbosity = 1, random_state = RANDOM_STATE, early_stopping_rounds = 10)
xgb_model.fit(x_train_fit,y_train_fit, eval_set = [(x_train_eval,y_train_eval)])
print("best:",xgb_model.best_iteration)
print(f"Metrices train:\n\tAccuracy score: {accuracy_score(xgb_model.predict(x_train),y_train):.4f}\nMetrices test:\n\tAccuracy score: {accuracy_score(xgb_model.predict(x_val),y_val):.4f}" )
