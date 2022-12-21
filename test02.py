import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn import  metrics

data = pd.read_csv("data/NBA.csv")
data.dropna(inplace=True)

X, y = data[['AW','AL','BW','BL','A10','B10']], data['SC']

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=1234)

reg=RandomForestRegressor(max_depth=2, random_state=0)
reg.fit(X_train, y_train)


print("Accuracy score", metrics.accuracy_score(y_test, reg.predict(y_train)))