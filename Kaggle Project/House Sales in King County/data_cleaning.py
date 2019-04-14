import hskc_functions
import numpy as np
import pandas as pd
from sklearn import preprocessing


def clean_up1(ds,rs):
    ds=ds.copy()
    del ds["id"]
    del ds["date"]
    del ds["sqft_living"]
    del ds["sqft_lot"]
    del ds["view"]
    del ds["lat"]
    del ds["long"]
    
    ds["age"] = 2018- np.where(ds["yr_renovated"]==0,ds["yr_built"],ds["yr_renovated"])
    del ds["yr_built"]
    del ds["yr_renovated"]
    ds["bathrooms"]=ds["bathrooms"]*ds["bedrooms"]
    ds=hskc_functions.createGroups(ds,"bedrooms")
    ds=hskc_functions.createGroups(ds,"bathrooms")
    ds=hskc_functions.createGroups(ds,"floors")
    ds=hskc_functions.dummifyField(ds,"zipcode")
    
    ds= hskc_functions.createQuantile(ds,"sqft_living15")
    #ds= createQuantile(ds,"sqft_above")
    
    result=hskc_functions.runRegression(ds)
    result.update({"Clean Up Id":"1","Comments":"Removed Id, date, sqft_living, sqft_lot,view, lat, long, yr_built, year_renovated\n"+
                   "Created age\n"+ 
                   "Updated bathrooms=bathrooms*bedrooms\nGrouped bedrooms,bathrooms,floors,\n"+
                   "Dummify zipcode\n"+
                   "Quantiled sqft_living15"})
    rs=rs.append(result,ignore_index=True)
    return ds,rs

def clean_up2(ds,rs):
    ds=ds.copy()
    del ds["id"]
    del ds["date"]
    del ds["sqft_living"]
    del ds["sqft_lot"]
    del ds["view"]
    del ds["lat"]
    del ds["long"]
    
    #Making zipcode as cities
    zc = pd.read_csv("C:\\Users\\manoj\\Documents\\Acadgild DSB\\Kaggle Project\\House Sales in King County\\Data\\zipcode.csv")
    ds = ds.merge(zc[["zipcode","City"]],on="zipcode")
    del ds["zipcode"]
    ds=hskc_functions.dummifyField(ds,"City")
    
    ds["age"] = 2018- np.where(ds["yr_renovated"]==0,ds["yr_built"],ds["yr_renovated"])
    del ds["yr_built"]
    del ds["yr_renovated"]
    ds["bathrooms"]=ds["bathrooms"]*ds["bedrooms"]
    ds=hskc_functions.createGroups(ds,"bedrooms")
    ds=hskc_functions.createGroups(ds,"bathrooms")
    ds=hskc_functions.createGroups(ds,"floors")
    #ds=hskc_functions.dummifyField(ds,"zipcode")
    
    ds= hskc_functions.createQuantile(ds,"sqft_living15")
    #ds= createQuantile(ds,"sqft_above")
    
    result=hskc_functions.runRegression(ds)
    result.update({"Clean Up Id":"2","Comments":"Removed Id, date, sqft_living, sqft_lot,view, lat, long, yr_built, year_renovated, zipcode\n"+
                   "Append City\n"+
                   "Created age\n"+ 
                   "Updated bathrooms=bathrooms*bedrooms\nGrouped bedrooms,bathrooms,floors,\n"+
                   "Dummify City\n"+
                   "Quantiled sqft_living15"})
    rs=rs.append(result,ignore_index=True)
    return ds,rs


def clean_up3(ds,rs):
    ds=ds.copy()
    del ds["id"]
    del ds["date"]
    del ds["sqft_living"]
    del ds["sqft_lot"]
    del ds["view"]
    del ds["lat"]
    del ds["long"]
    
    ds["age"] = 2018- np.where(ds["yr_renovated"]==0,ds["yr_built"],ds["yr_renovated"])
    del ds["yr_built"]
    del ds["yr_renovated"]
    ds["bathrooms"]=ds["bathrooms"]*ds["bedrooms"]
    ds=hskc_functions.createGroups(ds,"bedrooms")
    ds=hskc_functions.createGroups(ds,"bathrooms")
    ds=hskc_functions.createGroups(ds,"floors")
    ds=hskc_functions.dummifyField(ds,"zipcode")
    
    ds= hskc_functions.createQuantile(ds,"sqft_living15")
    
    
    
    #Scaling entire data
    x = ds.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    ds = pd.DataFrame(x_scaled)
    #ds= createQuantile(ds,"sqft_above")
    
    result=hskc_functions.runRegression(ds,True)
    result.update({"Clean Up Id":"3","Comments":"Removed Id, date, sqft_living, sqft_lot,view, lat, long, yr_built, year_renovated\n"+
                   "Created age\n"+ 
                   "Updated bathrooms=bathrooms*bedrooms\nGrouped bedrooms,bathrooms,floors,\n"+
                   "Dummify zipcode\n"+
                   "Quantiled sqft_living15\nData is scaled"})
    rs=rs.append(result,ignore_index=True)
    return ds,rs


def clean_up4(ds,rs):
    ds=ds.copy()
    del ds["id"]
    del ds["date"]
    del ds["sqft_living"]
    del ds["sqft_lot"]
    del ds["view"]
    del ds["lat"]
    del ds["long"]
    
    ds["age"] = 2018- np.where(ds["yr_renovated"]==0,ds["yr_built"],ds["yr_renovated"])
    del ds["yr_built"]
    del ds["yr_renovated"]
    ds["bathrooms"]=ds["bathrooms"]*ds["bedrooms"]
    ds=hskc_functions.createGroups(ds,"bedrooms")
    ds=hskc_functions.createGroups(ds,"bathrooms")
    ds=hskc_functions.createGroups(ds,"floors")
    ds=hskc_functions.dummifyField(ds,"zipcode")
    
    ds= hskc_functions.createQuantile(ds,"sqft_living15")
    #ds= createQuantile(ds,"sqft_above")
    
    result=hskc_functions.runRegression(ds)
    result.update({"Clean Up Id":"4","Comments":"Removed Id, date, sqft_living, sqft_lot,view, lat, long, yr_built, year_renovated\n"+
                   "Created age\n"+ 
                   "Updated bathrooms=bathrooms*bedrooms\nGrouped bedrooms,bathrooms,floors,\n"+
                   "Dummify zipcode\n"+
                   "Quantiled sqft_living15"})
    rs=rs.append(result,ignore_index=True)
    return ds,rs


def clean_up5(ds,rs):
    ds=ds.copy()
    del ds["id"]
    del ds["date"]
    del ds["sqft_living"]
    del ds["sqft_lot"]
    del ds["view"]
    del ds["lat"]
    del ds["long"]
    
    ds["age"] = 2018- np.where(ds["yr_renovated"]==0,ds["yr_built"],ds["yr_renovated"])
    del ds["yr_built"]
    del ds["yr_renovated"]
    ds["bathrooms"]=ds["bathrooms"]*ds["bedrooms"]
    ds=hskc_functions.createGroups(ds,"bedrooms")
    ds=hskc_functions.createGroups(ds,"bathrooms")
    ds=hskc_functions.createGroups(ds,"floors")
    ds=hskc_functions.dummifyField(ds,"zipcode")
    
    ds= hskc_functions.createQuantile(ds,"sqft_living15")
    #ds= createQuantile(ds,"sqft_above")
    
    #Scaling entire data
    x = ds.values #returns a numpy array
    standard_scalar = preprocessing.StandardScaler()
    x_scaled = standard_scalar.fit_transform(x)
    ds = pd.DataFrame(x_scaled)
    
    result=hskc_functions.runRegression(ds,True)
    result.update({"Clean Up Id":"5","Comments":"Removed Id, date, sqft_living, sqft_lot,view, lat, long, yr_built, year_renovated\n"+
                   "Created age\n"+ 
                   "Updated bathrooms=bathrooms*bedrooms\nGrouped bedrooms,bathrooms,floors,\n"+
                   "Dummify zipcode\n"+
                   "Quantiled sqft_living15"})
    rs=rs.append(result,ignore_index=True)
    return ds,rs






def clean_up6(ds,rs):
    ds=ds.copy()
    del ds["id"]
    del ds["date"]
    del ds["sqft_living"]
    del ds["sqft_lot"]
#    del ds["lat"]
#    del ds["long"]
    
    ds["age"] = 2018- np.where(ds["yr_renovated"]==0,ds["yr_built"],ds["yr_renovated"])
    del ds["yr_built"]
    del ds["yr_renovated"]
    
    ds["total_sqft"] = ds["sqft_above"] + ds["sqft_basement"]
    del ds["sqft_above"] 
    del ds["sqft_basement"]
    
    
#    ds= hskc_functions.createQuantile(ds,"sqft_living15")
#    ds= hskc_functions.createQuantile(ds,"sqft_lot15")
#    ds= hskc_functions.createQuantile(ds,"total_sqft")
    
    
    #result=hskc_functions.runRegression(ds)
    result=hskc_functions.runPolyRegression(ds)
    
    result.update({"Clean Up Id":"6","Comments":"Removed Id, date, sqft_living, sqft_lot, yr_built, year_renovated,sqft_above,sqft_basement\n"+
                   "Created age,total_sqft\n"+
                   "Quantiled sqft_living15,sqft_lot15,total_sqft"})
    rs=rs.append(result,ignore_index=True)
    return ds,rs


def clean_up7(ds,rs):
    ds=ds.copy()
    del ds["id"]
    del ds["date"]
    del ds["sqft_living"]
    del ds["sqft_lot"]
    del ds["lat"]
    del ds["long"]
    
#    ds["age"] = 2018- np.where(ds["yr_renovated"]==0,ds["yr_built"],ds["yr_renovated"])
#    del ds["yr_built"]
#    del ds["yr_renovated"]
    
#    ds["total_sqft"] = ds["sqft_above"] + ds["sqft_basement"]
#    del ds["sqft_above"] 
#    del ds["sqft_basement"]
    
#    ds=hskc_functions.dummifyField(ds,"bedrooms")
#    ds=hskc_functions.dummifyField(ds,"bathrooms")
#    ds=hskc_functions.dummifyField(ds,"floors")
#    ds=hskc_functions.dummifyField(ds,"view")
#    ds=hskc_functions.dummifyField(ds,"condition")
    ds=hskc_functions.dummifyField(ds,"grade")
    ds=hskc_functions.dummifyField(ds,"zipcode")
#    ds= hskc_functions.createQuantile(ds,"sqft_living15")
#    ds= hskc_functions.createQuantile(ds,"sqft_lot15")
#    ds= hskc_functions.createQuantile(ds,"total_sqft")
    
    
    #result=hskc_functions.runRegression(ds)
    result=hskc_functions.runPolyRegression(ds)
    
    result.update({"Clean Up Id":"7","Comments":"Removed Id, date, sqft_living, sqft_lot, yr_built, year_renovated,sqft_above,sqft_basement\n"+
                   "Created age,total_sqft\n"+
                   "Quantiled sqft_living15,sqft_lot15,total_sqft"})
    rs=rs.append(result,ignore_index=True)
    return rs



def clean_up8(ds,rs):
    ds=ds.copy()
    del ds["id"]
    del ds["date"]
    del ds["sqft_living"]
    del ds["sqft_lot"]
    del ds["lat"]
    del ds["long"]
    del ds["floors"]
    del ds["bedrooms"]
    del ds["bathrooms"]
    
    
    ds["age"] = 2018- np.where(ds["yr_renovated"]==0,ds["yr_built"],ds["yr_renovated"])
#    del ds["yr_built"]
#    del ds["yr_renovated"]
    
#    ds["total_sqft"] = ds["sqft_above"] + ds["sqft_basement"]
#    del ds["sqft_above"] 
#    del ds["sqft_basement"]
    ds=hskc_functions.makeFeaturePolyNomial(ds,"view")
    ds=hskc_functions.makeFeaturePolyNomial(ds,"condition")
    ds=hskc_functions.makeFeaturePolyNomial(ds,"sqft_living15")
    ds=hskc_functions.makeFeaturePolyNomial(ds,"sqft_lot15")
    ds=hskc_functions.makeFeaturePolyNomial(ds,"age")
#    ds=hskc_functions.dummifyField(ds,"bedrooms")
#    ds=hskc_functions.dummifyField(ds,"bathrooms")
#    ds=hskc_functions.dummifyField(ds,"floors")
#    ds=hskc_functions.dummifyField(ds,"view")
#    ds=hskc_functions.dummifyField(ds,"condition")
    ds=hskc_functions.dummifyField(ds,"grade")
    ds=hskc_functions.dummifyField(ds,"zipcode")
#    ds= hskc_functions.createQuantile(ds,"sqft_living15")
#    ds= hskc_functions.createQuantile(ds,"sqft_lot15")
#    ds= hskc_functions.createQuantile(ds,"total_sqft")
    
    
    result=hskc_functions.runRegression(ds)
    #result=hskc_functions.runPolyRegression(ds)
    
    result.update({"Clean Up Id":"8","Comments":"Removed Id, date, sqft_living, sqft_lot, yr_built, year_renovated,sqft_above,sqft_basement\n"+
                   "Created age,total_sqft\n"+
                   "Quantiled sqft_living15,sqft_lot15,total_sqft"})
    rs=rs.append(result,ignore_index=True)
    return ds,rs


def clean_up9(ds,rs):
    ds=ds.copy()
    del ds["id"]
    del ds["date"]

    

    
    
    result=hskc_functions.runRegression(ds)
    #result=hskc_functions.runPolyRegression(ds)
    
    result.update({"Clean Up Id":"9","Comments":"Removed Id, date,\n"+
                   "Created \n"+
                   "Quantiled "})
    rs=rs.append(result,ignore_index=True)
    return ds,rs