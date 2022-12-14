# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:16:16 2022

@author: lihsn
"""

import numpy as np


def choolesky(hessian, n, L, U):
    A=hessian
    info=0
    for i in range(n):
        for j in range(i+1):
            s=0
            for k in range(j):
                s+= U[i*n+k]* U[j*n+k]
                
            if i==j:
                if A[i*n+i]-s>=0:
                    U[i*n+j]=np.sqrt(A[i*n+i]-s)
                    L[j*n+i]=U[i*n+j]
                else:
                    info=1
            else:
                U[i*n+j]=1/U[j*n+j]*(A[i*n+j]-s)
                L[j*n+i]=U[i*n+j]
                
    return info, L, U