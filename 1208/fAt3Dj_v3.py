# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:02:22 2022

@author: lihsn
"""

import numpy as np


def fAt3Dj_v3(xc, yc, zc,xsize,ysize,zsize,delta_f,coeff):
    
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
    
    return temp