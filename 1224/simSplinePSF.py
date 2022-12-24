# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:41:51 2022

@author: e0947330
"""

import numpy as np
import math
from computeDelta3Dj_v2 import computeDelta3Dj_v2
from fAt3Dj_v2 import fAt3Dj_v2

def simSplinePSF(Npixels,coeff,I,bg,cor):
    
    Nfits = cor.shape[0]
    spline_xsize = coeff.shape[0]
    spline_ysize = coeff.shape[1]
    spline_zsize =coeff.shape[2]
    off = math.floor(((spline_xsize+1)-Npixels)/2)
    data = np.zeros((Npixels,Npixels,Nfits))
    
    for kk in range(Nfits):
        xcenter = cor[kk,0]
        ycenter = cor[kk,1]
        zcenter = cor[kk,2]
        
        xc = -1*(xcenter - Npixels/2+0.5)
        yc = -1*(ycenter - Npixels/2+0.5)
        zc = zcenter - math.floor(zcenter)
        
        xstart = math.floor(xc)
        xc = xc - xstart
    
        ystart = math.floor(yc)
        yc = yc - ystart
    

        zstart = math.floor(zcenter)
        delta_f0,delta_dxf0,delta_ddxf0,delta_dyf0,delta_ddyf0,delta_dzf0,delta_ddzf0=computeDelta3Dj_v2(xc,yc,zc)
        for ii in range(Npixels):
            for jj in range(Npixels):
                 temp = fAt3Dj_v2(ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f0,coeff)
                 model = temp*I+bg
                 data[jj,ii,kk]=model
    
    return data
       
Npixels=33
I=20000
bg=5
cor=[13.5,13.8,16.7]
cor=np.column_stack(cor)

simpsf=simSplinePSF(Npixels,coeff_xy,I,bg,cor)
import matplotlib.pyplot as plt
plt.imshow(simpsf[:,:,0],origin='lower')