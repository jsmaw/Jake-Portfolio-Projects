#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:48:01 2024

@author: jakesmaw
"""

# IMPORTED LIBRARIES
import pandas as pd
import numpy as np

# IMPORTED DATA
df = pd.read_csv('/Users/jakesmaw/Desktop/loan_approval_dataset.csv',index_col=0)
X = df.drop(' loan_status',axis=1)
y = df[' loan_status']

# SET POSITIVE ATTRIBUTES TO 1 AND NEGATIVE ATTRIBUTES TO 0 TO PREP FOR ENCODING
X[' education'] = X[' education'].apply(lambda x: 1 if x == ' Graduate' else 0)
X[' self_employed'] = X[' self_employed'].apply(lambda x: 1 if x == ' Yes' else 0)
y = y.apply(lambda x: 1 if x == ' Approved' else 0)

# CHECKED COLUMNS AND NULL DATA
df.columns
df.info()
df.isnull().sum()

# ENCODED LABELED DATA AND THE DEPENDENT VARIABLE
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[' education'] = le.fit_transform(X[' education'])
X[' self_employed'] = le.fit_transform(X[' self_employed'])
y = le.fit_transform(y)

# CHECKED COLUMNS TO MAKE SURE POSITIVE ATTRIBUTES HAD VALUE OF 1
X[' education']
X[' self_employed']
X.columns

# SPLIT DATASET INTO TRAIN AND TEST SETS
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# COMPLETED FEATURE SCALING
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# TRAINED LOGISTIC MODEL ON TRAINING SET
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

# PREDICTED A NEW RESULT
classifier.predict(scaler.transform([[2,1,0,9600000,29900000,12,778,2400000,17600000,22700000,8000000]]))
classifier.predict_proba(scaler.transform([[2,1,0,9600000,29900000,12,778,2400000,17600000,22700000,8000000]]))

# PREDICTED TEST SET RESULTS
pred = classifier.predict(X_test)

# VIEWED PREDICTIONS VS REAL RESULTS
print(np.concatenate((pred.reshape(len(pred),1),y_test.reshape(len(y_test),1)),1))

# OUTPUTED CONFUSION MATRIX AND CLASSIFICATION REPORT
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

# RAN A RANDOM FOREST CLASSIFICATION MODEL TO COMPARE 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train,y_train)
pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

# RESULTED IN 98% ACCURACY SCORE
# TUNED HYPERPARAMETERS AND NUMBER OF ESTIMATORS TO SEE IF WE CAN GET HIGHER ACCURACY
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_estimator_
grid_search.best_params_

# INPUTTED BEST PARAMS INTO MODEL AND RAN IT
classifier = RandomForestClassifier(n_estimators=300,max_depth=20,min_samples_leaf=1,min_samples_split=2)
classifier.fit(X_train,y_train)
pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

# RESULTED IN STILL A 98% ACCURACY
# FALSE NEGATIVES STAYED THE SAME, BUT RESULTED IN 2 LESS FALSE POSITIVES
# RANDOM FOREST MODEL RESULTS WERE SLIGHTLY BETTER