# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 04:05:13 2020

@author: Xemarij
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 03:14:52 2020

@author: Xemarij

Template for Building Regression Models 
"""
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
y = np.reshape(y,(10,1))

#Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split<X, y, test_size = 0.2,random_state=0)
"""

#Feature Scaling (We need it for SVR model brcause it not included
# in the imported library)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
	
#Fitting the Regression library SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)



#Create your regressor here


#Predicting a new result 
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary' )
plt.show()

#Visualizing the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color ='blue') 
plt.title('Truth or Bluff (Regression Model)') 
plt.xlabel('Position level') 
plt.ylabel('Salary')
plt.show()





 
