# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from fbprophet import Prophet
df = pd.read_csv('IBM Monthly.csv')
df = df.drop(["Open","High","Low","Adj Close","Volume"], axis=1)
df = df.rename(columns={"Date": "ds","Close": "y"})
df = df.drop([df["ds"].count()-1], axis=0)
m = Prophet(weekly_seasonality=True)
m.fit(df)
future = m.make_future_dataframe(periods=12,freq="MS")
forecast = m.predict(future)