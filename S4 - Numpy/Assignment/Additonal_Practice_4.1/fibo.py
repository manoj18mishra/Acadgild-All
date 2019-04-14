# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:28:54 2018

@author: manoj
"""

def fib2(num):
    f=0
    l=[1,1]
    while f < num:
        i=len(l)-1
        j=i-1
        f=l[i]+l[j]
        if f <= num:
            l=l+[f]
    return l 