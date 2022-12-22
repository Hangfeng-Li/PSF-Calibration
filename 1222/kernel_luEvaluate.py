# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 16:41:10 2022

@author: e0947330
"""
import numpy as np

def kernel_luEvaluate(L,U,b,n,x):
	
	y= np.zeros((6,1))

	for i in range(n):
		y[i,0] = b[i,0]
		for j in range(i):
			y[i,0] -= L[j*n+i,0] * y[j,0]
		
		y[i,0] /= L[i*n+i,0]
	
	
	for i in range(n-1,-1,-1):
	
		x[i,0] = y[i,0]
		for j in range(i+1,n):
		
			x[i,0] -= U[j*n+i,0] * x[j,0]
		
		x[i,0] /= U[i*n+i,0]
	return x