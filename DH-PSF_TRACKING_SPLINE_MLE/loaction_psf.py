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


scharr=np.ones((30,30)) 




grad=signal.convolve2d(input0[10,:,:],scharr,boundary='symm',mode='same') 
plt.imshow(input0[10,:,:], origin='lower')
plt.imshow(grad, origin='lower')
GRAD=((grad>7000) & (grad<10000))
plt.imshow(GRAD, origin='lower')

text11=(np.multiply(input0[10,:,:],GRAD))
text12=np.array(text11,dtype=int)

from skimage import io
io.imsave('G:/tony lab/cx//change.tif', text12)
