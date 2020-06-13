# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:05:37 2020

@author: Yang
"""


print("test")
def numerical_diff(f,x):
    h = 10e-4
    return (f(x+h)-f(x-h))/2