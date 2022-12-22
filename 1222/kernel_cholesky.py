# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 16:34:33 2022

@author: e0947330
"""

def kernel_cholesky(A,n,L,U):
	info = 0
	for i in range(n):
		for j in range(i+1):
			s = 0
			for k in range(j):
				s += U[i * n + k,0] * U[j * n + k,0]

			if (i==j):
				if (A[i*n+i,0]-s>=0):
					U[i * n + j,0] = (A[i * n + i,0] - s)**0.5
					L[j*n+i,0]=U[i * n + j,0]
				
				else:
					info =1
					
				
			
			else:
				U[i * n + j,0] = (1.0 / U[j * n + j,0] * (A[i * n + j,0] - s))
				L[j*n+i,0]=U[i * n + j,0]
	
	return info,L,U