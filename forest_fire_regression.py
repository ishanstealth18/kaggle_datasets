import pandas as pd
import numpy as np

df = pd.read_csv('forestfires.csv')
print(df.info())

#print(df.head())

from matplotlib import pyplot as plt


# log transform target variable
df['area'] = np.log1p(df['area'])

# Do one hot encoding for Month and Day
encode_col = ['month', 'day']
for col in encode_col:
    temp_dummy = pd.get_dummies(df[col], drop_first=True, prefix=col, dtype=int)
    df = pd.concat([df, temp_dummy], axis=1)
    df = df.drop([col], axis=1)

X = df.drop(['area'], axis=1)
y = df['area']

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
y_pred = regression_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('Linear Regression : MSE: ', mse)
print('Linear Regression : RMSE: ', rmse)
print('Linear Regression : R2 score: ', r2)
print('Linear Regression : MAE: ', mae)


# Decision Tree
from sklearn.tree import DecisionTreeRegressor
decision_tree_model = DecisionTreeRegressor(max_depth=500, random_state=42)
decision_tree_model.fit(X_train, y_train)
dt_pred = decision_tree_model.predict(X_test)

dt_mse = mean_squared_error(y_test, dt_pred)
dt_rmse = root_mean_squared_error(y_test, dt_pred)
dt_r2 = r2_score(y_test, dt_pred)
dt_mae = mean_absolute_error(y_test, dt_pred)
print('\nDecision Tree Regression : MSE: ', dt_mse)
print('Decision Tree Regression : RMSE: ', dt_rmse)
print('Decision Tree Regression : R2 score: ', dt_r2)
print('Decision Tree Regression : MAE: ', dt_mae)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,      # Number of trees to build
    max_depth=500,          # Limits tree depth to control model size
    random_state=42,       # Ensures reproducible results
    n_jobs=-1)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = root_mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
print('\nRandom Forest Regression : MSE: ', rf_mse)
print('Random Forest Regression : RMSE: ', rf_rmse)
print('Random Forest Regression : R2 score: ', rf_r2)
print('Random Forest Regression : MAE: ', rf_mae)