import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import autosklearn.classification
cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=120)
import sklearn.metrics

data = pd.read_csv("data/NBA.csv")
data.dropna(inplace=True)

X, y = data[['AW','AL','BW','BL','A10','B10']], data['WL']

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=1234)

cls.fit(X_train, y_train)
print(cls.leaderboard(detailed = True, ensemble_only=True))
print(cls.score(X_train, y_train))
print(cls.score(X_test, y_test))
quit()
y_pred = cls.predict(X_test)

print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))
quit()
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# reg = RandomForestClassifier()
#reg = KNeighborsClassifier()
reg = svm.SVC()
paras = {'C': [0.1,0.2,0.3,0.05, 1], 'kernel':['linear', 'poly', 'rbf', 'sigmoid']} #], 'max_depth': [1,2,3,5]}
clf = GridSearchCV(reg, paras, verbose=3)
clf.fit(X_train, y_train)
print(clf.best_score_)
print(clf.best_params_)
print(clf.score(X_test, y_test))
#y_pred = clf.predict(X_train)
