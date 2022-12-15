# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 23:21:03 2022

@author: lihsn
"""
import numpy as np


def DerivativeSpline(xc,yc,zc,xsize,ysize,zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,coeff,theta,dudt):
    temp=0
    xc=np.max(xc,0)
    xc=np.min(xc,xsize-1)
    
    yc=np.max(yc,0)
    yc=np.min(yc,ysize-1)
    
    zc=np.max(zc,0)
    zc=np.min(zc,zsize-1)
    
    for i in range(64):
        temp+=delta_f[i]*coeff[i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc]
        dudt[0]+=delta_dxf[i]*coeff[i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc]
        dudt[1]+=delta_dyf[i]*coeff[i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc]
        dudt[2]+=delta_dzf[i]*coeff[i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc]
    dudt[0]*=-1*theta[3];
    dudt[1]*=-1*theta[3]
    dudt[2]*=theta[3]
    dudt[3]=temp
    dudt[4]=1
    model = theta[4]+theta[3]*temp
    
    return dudt,model
    