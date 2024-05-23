#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:16:41 2024

@author: jakesmaw
"""

# IMPORTED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# IMPORTED DATA
df = pd.read_csv('/Users/jakesmaw/Desktop/data.csv',index_col=0)
df.info()
X = df.drop('diagnosis',axis=1)
y = df['diagnosis']

# CHECKED FOR NULL VALUES
df.isnull().sum()

# ENCODED DEPENDENT VARIABLE
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y

# SPLIT DATA INTO TRAIN AND TEST SETS
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# COMPLETED FEATURE SCALING
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train

# TRAINED KNN MODEL ON TRAINING SET
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train,y_train)

# PREDICTED TEST SET RESULTS
pred = classifier.predict(X_test)

# VIEWED PREDICTIONS VS REAL RESULTS
print(np.concatenate((pred.reshape(len(pred),1),y_test.reshape(len(y_test),1)),1))

# CONFUSION MATRIX AND CLASSIFICATION REPORT
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

# WILL FIND IDEAL NUMBER OF NEIGHBORS TO GET BETTER ACCURACY, CURRENTLY 95%
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i= knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure()
plt.plot(range(1,40),error_rate,marker='o')
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# 7 WAS FOUND TO BE IDEAL NUMBER OF NEIGHBORS
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train,y_train)
pred = classifier.predict(X_test)
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

# ACCURACY IS NOW 96%






















