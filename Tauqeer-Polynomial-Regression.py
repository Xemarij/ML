# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:35:37 2020
Polynomial Regression
@author: Xemarij
"""
#Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# When dataset is small, we do not have to split it in training and test sets.
#In this case we won't split it into training and testing sets.

# We also do not need to do Feature Scalling thanks to the libraries we will use.

#We will import LinearRegression lib to compare Linear and Polynomial Regression.
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Building poynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
#Now we will transform the poly_reg with following code
X_poly = poly_reg.fit_transform(X)
#We will create lin_reg_2 to fit our X_poly object to fit in our linear regression model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#We have built Linear Regression and Polynomial regression models for comparision.
#Now we will build the visulaision of Linear Regression results.
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('position level 1')
plt.ylabel('Salary 1')
plt.show()

#The linear model is not a good fit therefore we see the visualization for the polynomial
X_grid = np.arange(min(X), max(X), 0.1)
#The X_grid gives a smoother fit
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('position level')
plt.ylablel('Salary')
plt.show()

"""Use the following command in the lin_reg.predict(np.array([6.5]).reshape(1, 1))
the input pane to find the salary at level 6.5. This command is important and
can be used to predict salary of any level.
"""
"""Similarly, to predict with the more accurate polynomial regression for any level
use the following code.The level 6.5 is used as an example. Just replace 6.5 with any level value
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
"""




