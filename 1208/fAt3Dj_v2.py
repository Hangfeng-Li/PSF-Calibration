# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:36:13 2022

@author: e0947330
"""

import numpy as np


def fAt3Dj_v2(xc, yc, zc,xsize,ysize,zsize,delta_f,coeff):
    
    # spline_xsize,spline_ysize,spline_zsize
    
    # xsize=spline_xsize
    # ysize=spline_ysize
    # zsize=spline_zsize
    
    
    
    xc=max(xc,0)
    xc=min(xc,xsize-1)
    
    yc=max(yc,0)
    yc=min(yc,ysize-1)
    
    zc=max(zc,0)
    zc=min(zc,zsize-1)
    
    temp=coeff[int(xc),int(yc),int(zc),:]
    pd=np.sum(delta_f*temp)
    return pd