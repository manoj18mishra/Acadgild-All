# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:45:53 2018

@author: manoj
"""

import pandas as pd
import matplotlib.pyplot as plt

ds = pd.read_csv("https://stats.idre.ucla.edu/stat/data/binary.csv")
ds.hist()
plt.show()