import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score

#import Main
#from sklearn.preprocessing import StandardScaler
targetField = "affair"


def showGraph(ds):
    #ds[targetField].plot(kind="density")
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
    sns.distplot(ds[targetField],ax=ax[0])
    sns.distplot(ds["sqft_living"],ax=ax[1])
    
def runRegression(ds,scaled=False):
    if scaled:
        y=ds[0]
        x=ds.drop(columns=[0])
    else:
        y=ds[targetField]
        x=ds.drop(columns=[targetField])
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.20,random_state=1234)
    train_x.shape,test_x.shape,train_y.shape,test_y.shape
    
    model1=LinearRegression()
    model1.fit(train_x,train_y)
    train_r_squared=np.round(model1.score(train_x,train_y),2)
    print("Training Score --> ",train_r_squared)
    test_r_squared= np.round(model1.score(test_x,test_y),2)
    print("Test Score --> ",test_r_squared)
    #y_pred=model1.predict(test_x)
    #adjusted_r_squared = np.round( (1 - (1-test_r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1)),2)
    ARS_Train = np.round(AdjustedRSquare(model1,train_x,train_y),2)
    ARS_Test = np.round(AdjustedRSquare(model1,test_x,test_y),2)
    print("Adj R^2 Train --> ", ARS_Train)
    print("Adj R^2 Test --> ", ARS_Test)
    result = {"Clean Up Id":"","Training Score":train_r_squared,"Test Score":test_r_squared,"Adjusted R^2 Train":ARS_Train,"Adjusted R^2 Test":ARS_Test,"Comments":""}
    return result


def runLogisticsRegression(ds):
    y=ds[targetField]
    x=ds.drop(columns=[targetField])
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.20)
    train_x.shape,test_x.shape,train_y.shape,test_y.shape
    
    model1=LogisticRegression()
    model1.fit(train_x,train_y)
    train_r_squared=np.round(model1.score(train_x,train_y),2)
    print("Training Score --> ",train_r_squared)
    test_r_squared= np.round(model1.score(test_x,test_y),2)
    print("Test Score --> ",test_r_squared)
    #y_pred=model1.predict(test_x)
    #adjusted_r_squared = np.round( (1 - (1-test_r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1)),2)
    ARS_Train = np.round(AdjustedRSquare(model1,train_x,train_y),2)
    ARS_Test = np.round(AdjustedRSquare(model1,test_x,test_y),2)
    print("Adj R^2 Train --> ", ARS_Train)
    print("Adj R^2 Test --> ", ARS_Test)
    result = {"Clean Up Id":"","Training Score":train_r_squared,"Test Score":test_r_squared,"Adjusted R^2 Train":ARS_Train,"Adjusted R^2 Test":ARS_Test,"Comments":""}
    return result

def makeFeaturePolyNomial(ds,fieldName):
    poly = PolynomialFeatures(2)
    Xpoly = poly.fit_transform(ds[fieldName].values.reshape(-1,1))
    Xds=pd.DataFrame(Xpoly)
    ds[fieldName+"^2"]=Xds[2]
    return ds

def runPolyRegression(ds):
    y=ds[targetField]
    x=ds.drop(columns=[targetField])
    poly = PolynomialFeatures(2)
    Xpoly = poly.fit_transform(x)
    x = Xpoly
    
    
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.40,random_state=12)
    train_x.shape,test_x.shape,train_y.shape,test_y.shape
    
    
    #final_model = fit_model(train_x,train_y)
    
    #model1 = Lasso(alpha=10)
    
    model1=LinearRegression()
    model1.fit(train_x,train_y)
    train_r_squared=np.round(model1.score(train_x,train_y),2)
    print("Training Score --> ",train_r_squared)
    test_r_squared= np.round(model1.score(test_x,test_y),2)
    print("Test Score --> ",test_r_squared)
    #y_pred=model1.predict(test_x)
    #adjusted_r_squared = np.round( (1 - (1-test_r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1)),2)
    ARS_Train = np.round(AdjustedRSquare(model1,train_x,train_y),2)
    ARS_Test = np.round(AdjustedRSquare(model1,test_x,test_y),2)
    print("Adj R^2 Train --> ", ARS_Train)
    print("Adj R^2 Test --> ", ARS_Test)
    result = {"Clean Up Id":"","Training Score":train_r_squared,"Test Score":test_r_squared,"Adjusted R^2 Train":ARS_Train,"Adjusted R^2 Test":ARS_Test,"Comments":""}
    return result

def AdjustedRSquare(model,X,Y):
    YHat = model.predict(X)
    n,k = X.shape
    sse = np.sum(np.square(YHat-Y),axis=0)
    sst = np.sum(np.square(Y-np.mean(Y)),axis=0)
    R2 = 1- sse/sst
    adjR2 = R2-(1-R2)*(float(k)/(n-k-1))
    return adjR2

def fit_model(x,y):
    cv_sets = ShuffleSplit(n_splits=5,
                           test_size=0.20,
                           random_state=1234)
    ridgeModel = Ridge()
    params = {'alpha':list(range(0,5)),
             'solver' : ('auto', 
                         'svd', 
                         'cholesky', 
                         'lsqr', 
                         'sparse_cg', 
                         'sag', 
                         'saga')}
    scoring_func = make_scorer(performance_metric)
    grid = GridSearchCV(ridgeModel,params,scoring_func,cv=cv_sets)
    grid = grid.fit(x,y)
    return grid.best_estimator_

def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score
    
def createGroups(ds,fieldName):
    output = pairwise_tukeyhsd(ds[targetField],ds[fieldName])
    #output.plot_simultaneous()[0]
    d = output.summary()
    d = pd.DataFrame(d.data[1:],columns=d.data[0])
    o = getSimillarGroups(d)
    ds[fieldName]=ds[fieldName].map(o)
    ds=dummifyField(ds,fieldName)
    return ds

def createQuantile(ds,fieldName):
    dd = pd.qcut(ds[fieldName],q=[0, .25, .5, .75, 1.])
    dd = pd.Categorical(dd)
    dd = pd.get_dummies(dd,prefix=fieldName,drop_first=True)
    ds = pd.concat([ds.drop(columns=fieldName),dd],axis=1)
    return ds

def dummifyField(ds,fieldName):
    ds_dum = pd.get_dummies(ds[fieldName],prefix=fieldName,drop_first=True)
    ds = pd.concat([ds.drop(columns=fieldName),ds_dum],axis=1)
    return ds

def getSimillarGroups(ds):
    gu_s=ds.group1.append(ds.group2).unique()
    #ds=ds[~ds.reject]
    ds=ds[ds.reject]
    #o = pd.DataFrame(columns=["Val","Rep_Val"])
    o={}
    
    #for gu in ds.group1.append(ds.group2).unique():
    for gu in gu_s:
        x = ds[((ds.group1==gu) | (ds.group2==gu)) & (ds.meandiff.abs() == ds[(ds.group1==gu) | (ds.group2==gu)].meandiff.abs().min())]
        if x.shape[0]>0:
            #o = o.append({"Val":x["group1"].iloc[0],"Rep_Val":str(x["group1"].iloc[0])+"_"+str(x["group2"].iloc[0])},ignore_index=True)
            #o = o.append({"Val":x["group2"].iloc[0],"Rep_Val":str(x["group1"].iloc[0])+"_"+str(x["group2"].iloc[0])},ignore_index=True)
            o[x["group1"].iloc[0]]=str(x["group1"].iloc[0])+"_"+str(x["group2"].iloc[0])
            o[x["group2"].iloc[0]]=str(x["group1"].iloc[0])+"_"+str(x["group2"].iloc[0])
        #ds = ds[~((ds.group1==gu) | (ds.group2==gu))]
            ds = ds[~((ds.group1==x["group1"].iloc[0]) | (ds.group2==x["group1"].iloc[0]))]
            ds = ds[~((ds.group1==x["group2"].iloc[0]) | (ds.group2==x["group2"].iloc[0]))]
        else:
            if gu not in o:
                o[gu]=gu
        ds = ds[~((ds.group1==gu) | (ds.group2==gu))]
    return o