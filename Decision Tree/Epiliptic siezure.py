# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 11:41:09 2018

@author: manoj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#matplotlib inline

df = pd.read_csv("epilepsy.csv")
df['y'].value_counts()

del df['Unnamed: 0']

y=df['y']
x=df.drop(columns=["y"])

model_lr = LogisticRegression()
model_lr.fit(x,y)
model_lr.score(np.array(x),np.array(y))

from sklearn.decomposition import PCA
pcs = PCA(n_components=33)  #Keep changing until you get evr close to 90
pcs._fit(x)
pcs.explained_variance_
evr=pcs.explained_variance_ratio_
np.sum(evr)
pincomp=pcs.components_
pincomp.shape
scoring_matrix = pcs.transform(x)
scoring_matrix

model_lr_1 = LogisticRegression()
model_lr_1.fit(scoring_matrix,y)
model_lr_1.score(scoring_matrix,y)