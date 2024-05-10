#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:44:16 2024

@author: jakesmaw
"""

# THIS IS BANKRUPTCY DATA FROM THE TAIWAN ECONOMIC JOURNAL FROM 1999-2009
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/jakesmaw/Desktop/data.csv') 
df.head()
df.info()

# SCALED DATA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
df

# CHECKED PERCENT OF BANKRUPT COMPANIES
df['Bankrupt?'].value_counts(normalize=True)

# SPLIT TRAIN AND TEST SET
from sklearn.model_selection import train_test_split
X = df.drop('Bankrupt?',axis=1)
y = df['Bankrupt?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.model_selection import GridSearchCV

# DECIDED TO USE RANDOMFORESTCLASSIFIER MODEL
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
param_grid = {'n_estimators': range(10,100,10),'max_depth': range(10,100,10),"min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]}
grid_search = GridSearchCV(rfc,param_grid = param_grid,cv=5)
grid_search.fit(X_train,y_train)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
pred = best_model.predict(X_test)

# EVALUATED THE MODEL
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

# VISUALIZED RESULTS TO SHOW TOP 10 FEATURES TO PREDICT BANKRUPTCY
features = X_test.columns
importances = best_model.feature_importances_
feat_importance = pd.Series(importances , index=features).sort_values()
feat_importance.tail(10).plot(kind= 'barh')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")