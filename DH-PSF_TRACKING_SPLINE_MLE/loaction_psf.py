# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 16:54:38 2022

@author: e0947330
"""
import tifffile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

input0 = np.array(tifffile.imread('G:/tony lab/cx/E 488 off 561 on 001-half.tif'))
input0_size = input0.shape

scharr=np.ones((40,40)) 




grad=signal.convolve2d(input0[10,:,:],scharr,boundary='symm',mode='same') 
plt.imshow(input0[10,:,:], origin='lower')
plt.imshow(grad, origin='lower')