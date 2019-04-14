# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 11:36:25 2018

@author: manoj
"""

from sklearn.datasets import load_digits
digitData = load_digits()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()

X = digitData['data']
Y = digitData['target']

print (set(Y))


XTrain, XTest, YTrain, YTest = train_test_split(X,Y,test_size=0.25,
                                                random_state=1234)


XTrain.shape, XTest.shape, YTrain.shape, YTest.shape

from sklearn.linear_model import LogisticRegression

modelDigit = LogisticRegression(penalty='l1')
modelDigit.fit(XTrain,YTrain)

modelDigit.score(XTrain,YTrain)

modelDigit.score(XTest,YTest)

from sklearn import metrics
preds = modelDigit.predict(XTest)

print (metrics.confusion_matrix(YTest,preds))


print (metrics.classification_report(YTest, preds))

log_model

modelDigit

# AUC = 0.8----1.0----Very good model
# AUC = 0.7 ----0.8 -- good model
# AUC = 0.5---0.7--- needs improvement
# AUC = < 0.5 ---BAD
