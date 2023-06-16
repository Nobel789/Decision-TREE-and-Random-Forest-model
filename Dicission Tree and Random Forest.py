#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 05:31:11 2023

@author: myyntiimac
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import data and assign to df

df = pd.read_csv("/Users/myyntiimac/Desktop/EMP SAL.csv")
df.head()

#Define dEpendent and Independent and remove unwanted column

X = df.iloc[:, 1:2].values
X

y = df.iloc[:,2].values
y


### Dicision tree regreesin
#Decision Tree Regression is a supervised machine learning algorithm used for regression tasks. 
#the algorithm creates a binary tree-like model to make predictions. The tree is built by recursively splitting the feature space into subsets based on the values of the input features
#During the training phase, the algorithm evaluates different splits and selects the one that minimizes the variance or mean squared error (MSE) within each resulting subset. 
# The default parameter are(criterion': 'mse', 'splitter': 'best', 'max_depth': None, 'min_samples_split': 2,
# 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0, 'max_features': None,
 #'random_state': None)
 
 ## Lets build model with fit with data and algo then predict at6.5 level
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion = 'mse',splitter = 'random',max_depth = None)
   
regressor.fit(X, y)

#prediction
y_pred = regressor.predict([[6.5]])


#Lets check the perfomance with actual and predicted value line
plt.scatter(X, y, color = 'red')
plt.plot(X,regressor.predict(X), color = 'green')
plt.title('Truth or bluff (Decision tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()

#Insight:1)criterion = 'friedman_mse',splitter = 'random', we find predicted value at 6.5 is 15000
# parameter tuning with (criterion = 'mae',splitter = 'best') , its same value prdiction 
#Lets try with more parameter, (criterion='mae', splitter='best', max_depth'= None,)is same 
#(criterion = 'mse',splitter = 'random',max_depth = None) its predict 200000, not expected

## Lets build the Random Forest where all the parameter is same but only difference is Tree number , by deafult tree no or n-estimator 300, but you can put range 100 to 10000
# we will try to first default 300 then less and more
from sklearn.ensemble import RandomForestRegressor 
reg = RandomForestRegressor(n_estimators = 100, random_state = 0)
reg.fit(X,y)

#prediction
y_predRF = reg.predict([[6.5]])
y_predRF

#Lets chck with actual dpoint
plt.scatter(X, y, color = 'red')
plt.plot(X,reg.predict(X), color = 'green')
plt.title('Truth or bluff (Decision tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()

#Insight:with default estimator 300 , the model predict 160333, it can be optimized and prediced line didnot capture all d.point
#with tree no decreasing-200, its predict  159650
#Lets try in crease tree no 500- its again 160 k
#with 100 tree, 158300
#this model best prediction in trr no  default and 500 , is 160k

# so comparing both DT and RF we  can conclude that RF (160k)better predictor than DT(150k)

‰

