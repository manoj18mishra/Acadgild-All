import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
import seaborn as sns
#import sqlite3 as db
#from pandasql import sqldf
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import r2_score
#import scipy as sc
#import hskc_functions
import data_cleaning
from sklearn.preprocessing import PolynomialFeatures


#zc = pd.read_html("https://www.zip-codes.com/county/wa-king.asp")
h = pd.read_csv("C:\\Users\\manoj\\Documents\\Acadgild DSB\\Kaggle Project\\House Sales in King County\\Data\\kc_house_data.csv",parse_dates=["date"])
results=pd.DataFrame(columns=["Clean Up Id","Training Score","Test Score","Adjusted R^2 Train","Adjusted R^2 Test","Comments"])
h_c1,results=data_cleaning.clean_up1(h,results)
h_c2,results=data_cleaning.clean_up2(h,results)
h_c3,results=data_cleaning.clean_up3(h,results)
h_c4,results=data_cleaning.clean_up4(h,results)
#h_c5,results=data_cleaning.clean_up5(h,results)
#h_c6,results=data_cleaning.clean_up6(h,results)
#results=data_cleaning.clean_up7(h,results)
#h_c8,results=data_cleaning.clean_up8(h,results)
#h_c9,results=data_cleaning.clean_up9(h,results)

#poly = PolynomialFeatures(2)
#Xpoly = poly.fit_transform(h["bedrooms"].values.reshape(-1,1))
#Xds=pd.DataFrame(Xpoly)
#Xds.columns=[1,2,"Test"]

#del h_c1
#del h_c2
#del h_c3
#del h_c4
#del h_c5

results.to_csv("results.csv")
#hskc_functions.showGraph(ht)

'''Finding duplicates based on date
ids = h["id"]
x=h[ids.isin(ids[ids.duplicated()])]
'''