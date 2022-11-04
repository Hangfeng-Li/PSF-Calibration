# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:41:50 2022

@author: e0947330
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PSF_generate_preprocess import PSF_generate_preprocess
from PSF_generate_image import PSF_generate_image


filename = 'G:/tony lab/cx/denoise/mask0.mat'
mask0 = loadmat(filename)
print(type(mask0))
mask = mask0['mask0']
plt.imshow(mask, origin='lower')


lamda=0.514#um
NA=1.45
M=100
f_4f=30e4#mm
pixel_size_CCD=6.5#um
pixel_size_SLM=9.2#um
n_glass=1.518
n_medium=1.33
k=2*math.pi/lamda
z_emit=0.01#emitter radius [um]
SAF_flag=1
polar_flag=3
vec_flag=1
angle_azimuth=0
angle_pitch=90
u=np.array([np.sin(angle_pitch)*np.cos(angle_azimuth),np.sin(angle_pitch)*np.sin(angle_azimuth),np.cos(angle_pitch)])
u=[0,0,0]
signal=1

x_position0=np.zeros(800)
y_position0=np.zeros(800)
z_position0=np.linspace(0,4,800)+z_emit
# x_position0=[0,0,0,0,0]
# y_position0=[0,0,0,0,0]
# z_position0=[z_emit,z_emit+1,z_emit+2,z_emit+3,z_emit+4]
x_position0=np.array(x_position0)
y_position0=np.array(y_position0)
z_position0=np.array(z_position0)

# NFP0=np.linspace(-3,3,5)
NFP0=np.ones(800)*(-2)
NFP0=np.array(NFP0)
FOV_size  =200



# data_stack=
# N_bg_stack=
# std_bg_stack=
# gBlur=
# cost_alter=np.ones((FOV_size,FOV_size,NFP0.shape[0]))
# Nph_opt_flag=1



d_mask_pixel,phi,N_crop,g_bfp,circ_mask_opt,N,circ_mask=PSF_generate_preprocess(lamda,NA,M,f_4f,pixel_size_CCD,pixel_size_SLM,n_glass,n_medium,k,z_emit,SAF_flag,polar_flag,vec_flag,u,signal,FOV_size)


I_img_stack=np.zeros((FOV_size,FOV_size))
I_img0=np.zeros((FOV_size,FOV_size,x_position0.shape[0]))
I_img1=np.zeros((FOV_size,FOV_size,x_position0.shape[0]))
for emit_num in range(x_position0.shape[0]):
    x_position=x_position0[emit_num]
    y_position=y_position0[emit_num]
    z_position=z_position0[emit_num]
    NFP=NFP0[emit_num]
    I_img0[:,:,emit_num]=PSF_generate_image(d_mask_pixel,phi,mask,N_crop,g_bfp,circ_mask_opt,N,circ_mask,n_glass,n_medium,k,z_emit,SAF_flag,polar_flag,vec_flag,u,signal,x_position,y_position,z_position,NFP,FOV_size)
    # I_img_stack=I_img_stack+I_img0[:,:,emit_num]
    I_img1[:,:,emit_num]=I_img0[:,:,emit_num]/np.sum(I_img0[:,:,emit_num])

np.save('G:/tony lab/cx/denoiseI_img1.npy',I_img1)
    
plt.imshow(I_img0[:,:,4], origin='lower')
  
    