# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:16:16 2022

@author: lihsn
"""

import numpy as np


def choolesky(hessian, n, L1, U1):
    A=hessian
    info=0
    for i in range(n):
        for j in range(i+1):
            s=0
            for k in range(j):
                # if U1[i*n+k]==0:
                #     U1[i*n+k]=0.1
                # if U1[j*n+k]==0:
                #     U1[j*n+k]=0.1
                s+= U1[i*n+k]* U1[j*n+k]
                
            if i==j:
                if A[i*n+i]-s>=0:
                    U1[i*n+j]=np.sqrt(A[i*n+i]-s)
                    L1[j*n+i]=U1[i*n+j]
                else:
                    info=1
            else:
                U1[i*n+j]=1/U1[j*n+j]*(A[i*n+j]-s)
                L1[j*n+i]=U1[i*n+j]
                
    return info,L1,U1


#     A=hessian.reshape([5,5])
#     A=A+A.T
#     B=np.linalg.eigvals(A)
#     if np.all(B>0):
#         info=0
#     else:
#         info=1
#     return info


# # U1=np.zeros((25,1))
# # L1=np.zeros((25,1))
# # n=NV

# # U2=U1.reshape([5,5])
# # L2=L1.reshape([5,5])

# # hessian1=hessian.reshape([5,5])
# # hessian1=hessian1+hessian1.T
# # hessian2=np.triu(hessian1)
