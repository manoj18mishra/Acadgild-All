# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:51:36 2018

@author: manoj
"""

def palindrome(s):
    hl=int(len(s)/2)
    if s.lower() == s[::-1].lower():
        print("{} is a palindrome".format(s))
    else:
        print("{} is not a palindrome".format(s))
        