import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/NBA.csv")
data.dropna(inplace=True)

X, y = data[['AW','AL','BW','BL','A10','B10']], data[['WL']]

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.2, random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

reg= RandomForestClassifier()
reg.fit(X_train, y_train)
ypred = reg.predict(X_test)

score = accuracy_score(y_test, ypred)
print(score)
# adiff=y_test.iloc[:, 0]-ypred[:, 0]
# bdiff=y_test.iloc[:, 1]-ypred[:, 1]
# plt.plot(np.arange(len(y_test)), y_test.iloc[:, 0],'o-')
# plt.plot(np.arange(len(y_test)), ypred[:, 0],'o-')
# plt.plot(np.arange(len(y_test)),adiff,'o-')
# plt.plot(np.arange(len(y_test)),bdiff,'o-')
# plt.legend(['AD','BD'])
# plt.grid()
# plt.show()