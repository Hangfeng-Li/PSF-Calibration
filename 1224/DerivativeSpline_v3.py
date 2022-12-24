# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 23:02:05 2022

@author: lihsn
"""

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


# 		temp+=delta_f[i,0]*coeff[int(i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc)]
# 		dudt[0,0]+=delta_dxf[i,0]*coeff[int(i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc)]
# 		dudt[1,0]+=delta_dyf[i,0]*coeff[int(i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc)]
# 		dudt[4,0]+=delta_dzf[i,0]*coeff[int(i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc)]
        
# #         temp+=delta_f[i,0]*coeff[int(i+zc*(xsize*ysize)*64+yc*xsize*64+xc*64)]
# # 		dudt[0,0]+=delta_dxf[i,0]*coeff[int(i+zc*(xsize*ysize)*64+yc*xsize*64+xc*64)]
# # 		dudt[1,0]+=delta_dyf[i,0]*coeff[int(i+zc*(xsize*ysize)*64+yc*xsize*64+xc*64)]
# # 		dudt[4,0]+=delta_dzf[i,0]*coeff[int(i+zc*(xsize*ysize)*64+yc*xsize*64+xc*64)]
# 	
# 	dudt[0,0]*=-1*theta[2,0]
# 	dudt[1,0]*=-1*theta[2,0]
# 	dudt[4,0]*=theta[2,0]
# 	dudt[2,0]=temp
# 	dudt[3,0]=1
# 	model = theta[3,0]+theta[2,0]*temp
    




def DerivativeSpline_v3(ii,jj,xc,yc,zc,xsize,ysize,zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,coeff,theta,dudt,off,xstart,ystart,zstart):
    temp=0
    # xc=max(xc,0)
    # xc=min(xc,xsize-1)
    
    # yc=max(yc,0)
    # yc=min(yc,ysize-1)
    
    # zc=max(zc,0)
    # zc=min(zc,zsize-1)
    dudt=np.zeros((5,1))
    
    temp0 = fAt3Dj_v3(ii+xstart+off,jj+ystart+off,zstart,xsize,ysize,zsize,delta_f,coeff)
    # pd=np.sum(delta_f*temp)
    for i in range(64):
        temp=delta_f[i,0]*temp0[i,0]+temp
        dudt[0,0]+=delta_dxf[i,0]*temp0[i,0]
        dudt[1,0]+=delta_dyf[i,0]*temp0[i,0]
        dudt[4,0]+=delta_dzf[i,0]*temp0[i,0]
    dudt[0,0]*=-1*theta[2,0]
    dudt[1,0]*=-1*theta[2,0]
    dudt[4,0]*=theta[2,0]
    dudt[2,0]=temp
    dudt[3,0]=1
    model = theta[3,0]+theta[2,0]*temp
    
    # dudt=np.zeros((5,1))
    
    # temp0,temp = fAt3Dj_v3(ii+xstart+off,jj+ystart+off,zstart,xsize,ysize,zsize,delta_f,coeff)
    
    # # pd=np.sum(delta_f*temp)
    
    
    # dudt[0,0]=np.sum(delta_dxf*temp)
    # dudt[1,0]=np.sum(delta_dyf*temp)
    # dudt[4,0]=np.sum(delta_dzf*temp)
    # dudt[0,0]*=-1*theta[2,0]
    # dudt[1,0]*=-1*theta[2,0]
    # dudt[4,0]*=theta[2,0]
    # dudt[2,0]=temp0
    # dudt[3,0]=1
    # model = theta[3,0]+theta[2,0]*temp0
    
    return model,dudt








