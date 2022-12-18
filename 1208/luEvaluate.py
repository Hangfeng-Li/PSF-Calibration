# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:32:03 2022

@author: lihsn
"""

import numpy as np


def luEvaluate(L, U, b, n, x):
    
    y=np.zeros((5,1))
    i=0
    j=0
    
    for i in range(n):
        y[i]=b[i]
        for j in range(i):
            y[i]-=L[j*n+i]*y[j]
        y[i]/=L[i*n+i]
        
    for i in range(n-1,-1,-1):
        x[i]=y[i]
        for j in range(i+1,n):
            x[i]-=U[j*n+i]*x[j]
        x[i]/=U[i*n+i]
        
    return x