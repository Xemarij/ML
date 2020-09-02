# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 18:28:28 2020

@author: Xemarij
"""
#Simple Linear Regression Preprocessing
#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Splitting te data into training set and test set
#The code in the coments is depricated. The code following it is valid. 
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Feature Scalling (We don't need to apply feature scalling on Simple Linear
# Regression problems. The library will take care of it)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

#Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Visualizing the training set results
plt.scatter(X_train, y_train,  color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Exprience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show() 