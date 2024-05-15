#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:11:47 2024

@author: jakesmaw
"""

# IMPORTED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# IMPORTED DATA
df = pd.read_csv('/Users/jakesmaw/Desktop/insurance.csv')
df.info()
X = df.iloc[:,:-1]
y = df.iloc[:,-1].values

# CHECKED IF NULL VALUES IN DATASET
df.isnull().sum()

# ENCODED CATEGORICAL DATA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
categorical_data = ['sex','smoker','region']
ct = ColumnTransformer(transformers=[('encoder',
        OneHotEncoder(),categorical_data)],remainder='passthrough')
X = np.array(ct.fit_transform(X))
X

# CONFIRM X HAS CORRECT NUMBER OF COLUMNS (11)
X.shape

# SPLIT DATA INTO TRAINING AND TESTING DATASETS
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# NO NEED TO STANDARDIZE NUMERICAL COLUMNS BECAUSE
# THESE ARE DECISION TREE AND RANDOM FOREST MODELS

# FIRST TRIED DECISION TREE REGRESSION MODEL
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

# PREDICT CHARGES FOR A MALE, WHO SMOKES, IN SOUTHWEST REGION 
# WHO IS 35 YEARS OLD, WITH BMI = 27.3, AND 2 CHILDREN
print(regressor.predict([[1,0,1,0,0,0,0,1,35,27.3,2]]))
# OUR MODEL SAYS HE WILL BE CHARGED $6402.29

# COMPARED TO RANDOM FOREST REGRESSION MODEL
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print(regressor.predict([[1,0,1,0,0,0,0,1,35,27.3,2]]))
# THE SAME MAN WAS GIVEN A PREDICTED CHARGED OF 
# BASED ON THE RANDOM FOREST $7960.99

# CHECK R2 VALUES FOR EACH MODEL
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
# DECISION TREE MODEL HAS R2 = 0.753
# RANDOM FOREST MODEL HAS R2 = 0.797

# WE WILL USE RANDOM FOREST MODEL BECAUSE IT PERFORMED BETTER
# NOW I AM GOING TO FIND THE BEST NUMBER OF ESTIMATORS AND OTHER HYPERPARAMTERS TO USE FOR MY MODEL
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50,100,200,250],'min_samples_leaf': [1, 2, 4],'max_depth': [None, 10, 20, 30]}
regressor = RandomForestRegressor()
grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_parameters

# THE BEST ESTIMATOR WAS 250 SO I REPLACED IT AND THE OTHER BEST HYPERPARAMTERS VALUES INTO MY MODEL
# I RE-RAN THE MODEL AND THIS IS THE OUTCOME
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=250,min_samples_leaf=4,max_depth=None)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# THE NEW R2 = 0.808, WHICH IS BETTER THAN THE ORIGINAL 0.753

