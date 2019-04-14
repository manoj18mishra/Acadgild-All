import pandas as pd
import numpy as np 

dataset=pd.read_csv('Data.csv')

#divide data into independent and dependent
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

#Handle missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy="mean",axis=0)
imputer=imputer.fit(X[:1,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
#Above 2 lines of code can be done in the following 1 code
#X[:,1:3]=imputer.fit_transform(X[:,1:3])


#Onehot encoding and label encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder() # can be used later for 3 values categorial of any column
X[:,0]=labelencoder_X.fit_transform(X[:,0]) 
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

#Dummy variable trap
X=X.iloc[:,1:]

#Divide dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_text=train_test_split(X,Y,test_size=0.2,random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train.reshape(-1,1))