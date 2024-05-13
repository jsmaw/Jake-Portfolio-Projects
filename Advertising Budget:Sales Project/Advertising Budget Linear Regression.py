#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:40:05 2024

@author: jakesmaw
"""


'''
THIS MODEL HELPS TO PREDICT WHAT THE SALES
WILL BE GIVEN THE TV, RADIO, AND NEWSPAPER AD BUDGETS
''' 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

df = pd.read_csv('/Users/jakesmaw/Desktop/Advertising Budget and Sales.csv')
df = df.drop(df.columns[0],axis=1)
df.info()
df.isnull().sum()

sns.scatterplot(df, x = "TV Ad Budget ($)", y = "Sales ($)")
sns.scatterplot(df, x = "Radio Ad Budget ($)", y = "Sales ($)")
sns.scatterplot(df, x = "Newspaper Ad Budget ($)", y = "Sales ($)")
sns.pairplot(df)

df.corr()

X = df.iloc[:,:-1].values
y = df['Sales ($)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train,y_train)
pred = regressor.predict(X_test)

print(regressor.predict([[200,40,25]]))

print(regressor.coef_)
print(regressor.intercept_)
# y = (regressor.coef_ * X) + regressor.intercept_

from sklearn import metrics
print('RMSE: ', np.sqrt(metrics.mean_absolute_error(y_test,pred)))
