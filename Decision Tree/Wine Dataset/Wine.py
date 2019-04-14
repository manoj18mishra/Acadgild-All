# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:26:19 2018

@author: manoj
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from  sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

ds_w = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",sep=';')
ds_r = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",sep=';')
ds = pd.concat([ds_w,ds_r])

# How to model immbalanced dataset
# which model performs best
# how to do benchmarking
# conclusion
ds.info()
ds['quality'] = pd.Categorical(ds['quality'])
ds.info()
set(ds.quality)
plt.figure(figsize=(10,6))
sns.distplot(ds["fixed acidity"])
plt.figure(figsize=(10,6))
sns.distplot(ds["volatile acidity"])
plt.figure(figsize=(10,6))
sns.boxplot(ds["fixed acidity"]) #just to check how outliers are handled by Decision Tree
X = ds.drop(columns=['quality']).as_matrix() # .as_matrix() or np.array will do the same thing
y = np.arrary(ds['quality'])
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.20,random_state=1234)
lm = LogisticRegression()
lm
lm.fit(Xtrain,Ytrain)
lm.predict(Xtest)
lm.score(Xtrain,Ytrain)
lm.score(Xtest,Ytest)

pred_quality = lm.predict(Xtest)
confusion_matrix(Ytest,pred_quality)
print(classification_report(Ytest,pred_quality))

lm2 = LogisticRegression()
def GridSearch_BestParam(X, y, clf, param_grid,cv=10):
    grid_search = GridSearchCV(clf,
                              param_grid=param_grid,
                              cv=cv)
    start= time()
    grid_search.fit(X,y)
    top_score=grid_search.best_score_
    top_params=grid_search.best_params_  #grid_search.grid_scores_ old new is best_score_
    return top_score,top_params

lm2
param_grid = {"C":[0.001,0.05,0.1],
              'solver':['newton-cg','lbfgs','liblinear']}
from sklearn.model_selection import GridSearchCV
from time import time
from operator import itemgetter
top_para = GridSearch_BestParam(Xtrain,Ytrain,lm2,param_grid,cv=10)
print(top_para)
top_para
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt
dt.fit(Xtrain,Ytrain)
dt.score(Xtrain,Ytrain)
dt.score(Xtest,Ytest)
param_grid={"criterion": ["gini","entropy"],
           "min_samples_split":[10,20],  #use min 30 in real project
           "max_depth": [2,5,7,10],
           "min_samples_leaf":[10]
           }
top_para = GridSearch_BestParam(Xtrain,Ytrain,dt,param_grid,cv=10)
print(top_para)
dt.feature_importances_
ds.columns