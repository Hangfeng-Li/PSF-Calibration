# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 16:45:48 2022

@author: e0947330
"""
import numpy as np
def kernel_DerivativeSpline(xc,yc,zc,xsize,ysize,zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,coeff,theta,dudt,model):
	temp =0
	
	
	xc = max(xc,0)
	xc = min(xc,xsize-1)

	yc = max(yc,0)
	yc = min(yc,ysize-1)

	zc = max(zc,0)
	zc = min(zc,zsize-1)
	dudt=np.zeros((5,1))
	

	for i in range(64):	
		temp+=delta_f[i,0]*coeff[int(i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc)]
		dudt[0,0]+=delta_dxf[i,0]*coeff[int(i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc)]
		dudt[1,0]+=delta_dyf[i,0]*coeff[int(i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc)]
		dudt[4,0]+=delta_dzf[i,0]*coeff[int(i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc)]
	
	dudt[0,0]*=-1*theta[2,0]
	dudt[1,0]*=-1*theta[2,0]
	dudt[4,0]*=theta[2,0]
	dudt[2,0]=temp
	dudt[3,0]=1
	model = theta[3,0]+theta[2,0]*temp
    
	return model,dudt
