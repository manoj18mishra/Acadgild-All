import functions
import numpy as np
import pandas as pd
from sklearn import preprocessing


def clean_up1(ds,rs):
    ds=ds.copy()
    ds["affairs time ratio"]=ds["affairs"]/ds["yrs_married"]
    del ds["affairs"]
    ds= functions.createQuantile(ds,"yrs_married")
    #ds=functions.createGroups(ds,"bedrooms")
    #ds=functions.dummifyField(ds,"zipcode")
    #ds= functions.createQuantile(ds,"sqft_living15")
    result=functions.runLogisticsRegression(ds)
    result.update({"Clean Up Id":"1","Comments":"Removed Id, date, sqft_living, sqft_lot,view, lat, long, yr_built, year_renovated\n"+
                   "Created age\n"+ 
                   "Updated bathrooms=bathrooms*bedrooms\nGrouped bedrooms,bathrooms,floors,\n"+
                   "Dummify zipcode\n"+
                   "Quantiled sqft_living15"})
    rs=rs.append(result,ignore_index=True)
    return ds,rs

