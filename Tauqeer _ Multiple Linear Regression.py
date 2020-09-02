# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 02:30:07 2020
Multiple Linear Regression
@author: Xemarij
"""

#Multiple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding categorical data (The following code in comments has been depricated)

""" 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_X = LabelEncoder()
X[:,3] =  LabelEncoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
"""

#The above code should be like in the following
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],    # The column index to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype=np.float)

# Manually avoiding the dummy variable trap by removing column with index 0.
#It should be noted that libraries can take care that for you automatically also.
X = X[:, 1:]



#Splitting the data into training set and test set
#The code in the coments is depricated. The code following it is valid. 
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scalling (We don't need to apply feature scalling on Simple Linear
# Regression problems. The library will take care of it)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Building the optimal model with backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis=1)

X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

#After eliminating the P values that are higher than the Significant Level of 0.05
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())