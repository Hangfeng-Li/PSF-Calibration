# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 23:21:03 2022

@author: lihsn
"""
import numpy as np


# xc=ii+xstart+off
# yc=jj+ystart+off
# zc=zstart
# xsize=spline_xsize
# ysize=spline_ysize
# zsize=spline_zsize
# theta=newtheta
# # coeff=spline_coeff
# dudt=np.zeros((5,1))


def DerivativeSpline(xc,yc,zc,xsize,ysize,zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,spline_coeff,theta,dudt:
    temp=0
    xc=max(xc,0)
    xc=min(xc,xsize-1)
    
    yc=max(yc,0)
    yc=min(yc,ysize-1)
    
    zc=max(zc,0)
    zc=min(zc,zsize-1)
    dudt=np.zeros((5,1))
    
    for i in range(64):
        temp+=delta_f[i]*spline_coeff[int(i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc)]
        dudt[0]+=delta_dxf[i]*spline_coeff[int(i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc)]
        dudt[1]+=delta_dyf[i]*spline_coeff[int(i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc)]
        dudt[2]+=delta_dzf[i]*spline_coeff[int(i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc)]
    dudt[0]*=-1*theta[3]
    dudt[1]*=-1*theta[3]
    dudt[2]*=theta[3]
    dudt[3]=temp
    dudt[4]=1
    model = theta[4]+theta[3]*temp
    
    return dudt,model

