# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 23:11:14 2022

@author: lihsn
"""

import numpy as np


def SplineMLE(box, data,spline_coeff,spline_xsize,spline_ysize,spline_zsize,size_box,iterations,init_parameters):
    
    NV=5
    M=np.zeros((NV*NV,1))
    Diag=np.zeros((NV,1))
    Minv=np.zeros((NV*NV))
    
    xstart=init_parameters[0]
    ystart=init_parameters[1]
    zstart=init_parameters[2]
    Nstart=init_parameters[3]
    Bgstart=init_parameters[4]
    
    
    xc=xstart
    yc=ystart
    zc=zstart
    
    
    
    newlambda=0.1
    oldlambda=0.1
    newupdate=[1e13,1e13,1e13,1e13,1e13]
    newupdate=np.column_stack(newupdate).T
    oldupdate=[1e13,1e13,1e13,1e13,1e13]
    oldupdate=np.column_stack(oldupdate).T
    maxjump=[1,1,2,100,20]
    maxjump=np.column_stack(maxjump).T
    newerr=1e12
    olderr=1e13
    
    newdudt=np.zeros((NV,1))
    jacobian=np.zeros((NV,1))
    delta_f=np.zeros((64,1))
    delta_dxf=np.zeros((64,1))
    delta_dyf=np.zeros((64,1))
    delta_dzf=np.zeros((64,1))
    errflag=0
    L=np.zeros((NV*NV,1))
    U=np.zeros((NV*NV,1))
    
    Nmax=np.max(data)
    newtheta=init_parameters
    newtheta[4]=np.max([newtheta[4],0.01])
    newtheta[3]=(Nmax-newtheta[3])/spline_coeff[(spline_zsize/2)*(spline_xsize*spline_ysize)+(spline_ysize/2)*spline_xsize+(spline_xsize/2)]*4
    #(x,y,bg,I,z)
    maxjump[4]=np.max([newtheta[4],maxjump[4]])
    maxjump[3]=np.max([newtheta[3],maxjump[3]])
    maxjump[2]=np.max([spline_zsize/3,maxjump[2]])
    
    oldtheta=newtheta
    
    xc=-1*(newtheta[0]-size_box/2+0.5)
    xc=-1*(newtheta[1]-size_box/2+0.5)
    
    off=(spline_xsize+1-size_box)/2
    
    xstart=xc
    xc=xc-xstart
    
    ystart=yc
    yc=yc-ystart
    
    zstart=newtheta[2]
    zc=newtheta[2]-zstart
    
    newerr=0
    
    ####带入xc，yc，zc等计算delta_f,delta_dxf,delta_dyf,delta_dzf
    
    #
    
    for ii in range(size_box):
        for jj in range(size_box):
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    