# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:03:04 2022

@author: lihsn
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 23:21:03 2022

@author: lihsn
"""
import numpy as np
from fAt3Dj_v3 import fAt3Dj_v3


# xc=ii+xstart+off
# yc=jj+ystart+off
# zc=zstart
# xsize=spline_xsize
# ysize=spline_ysize
# zsize=spline_zsize
# theta=newtheta
# # coeff=spline_coeff
# dudt=np.zeros((5,1))


def DerivativeSpline_v2(ii,jj,xc,yc,zc,xsize,ysize,zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,coeff,theta,dudt,off,xstart,ystart,zstart):
    temp=0
    xc=max(xc,0)
    xc=min(xc,xsize-1)
    
    yc=max(yc,0)
    yc=min(yc,ysize-1)
    
    zc=max(zc,0)
    zc=min(zc,zsize-1)
    dudt=np.zeros((5,1))
    
    temp0 = fAt3Dj_v3(ii+xstart+off,jj+ystart+off,zstart,xsize,ysize,zsize,delta_f,coeff)
    for i in range(64):
        temp+=temp0[i]
        dudt[0]+=delta_dxf[i]*temp0[i]
        dudt[1]+=delta_dyf[i]*temp0[i]
        dudt[2]+=delta_dzf[i]*temp0[i]
    dudt[0]*=-1*theta[3]
    dudt[1]*=-1*theta[3]
    dudt[2]*=theta[3]
    dudt[3]=temp
    dudt[4]=1
    model = theta[4]+theta[3]*temp
    
    return dudt,model


    
    
                 
            