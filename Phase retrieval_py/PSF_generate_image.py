# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:47:30 2022

@author: e0947330
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat

def PSF_generate_image(d_mask_pixel,phi,mask,N_crop,g_bfp,circ_mask_opt,N,circ_mask,n_glass,n_medium,k,z_emit,SAF_flag,polar_flag,vec_flag,u,signal,x_position,y_position,z_position,NFP,FOV_size):
    
    
    bfp_phase=np.zeros((int(d_mask_pixel),int(d_mask_pixel)))
    bfp_phase_exp=np.zeros((int(d_mask_pixel),int(d_mask_pixel)))+0*1j
    
        
    bfp_phase[:,:]=k*(z_position*phi[:,:,2]+x_position*phi[:,:,0]+y_position*phi[:,:,1]+NFP*phi[:,:,3])  
    bfp_phase[:,:][np.isnan(bfp_phase[:,:])]=0
    bfp_phase_exp[:,:]=np.exp(1j*(bfp_phase[:,:]+mask))
    
    I_img_stack=np.zeros((N_crop,N_crop))   
                         
    
        
    if vec_flag==1:
        g_img=np.zeros((int(d_mask_pixel),int(d_mask_pixel),g_bfp.shape[2]))+0*1j
        g_bfp_1=np.zeros((int(d_mask_pixel),int(d_mask_pixel),g_bfp.shape[2]))+0*1j
        I_img=np.zeros((int(d_mask_pixel),int(d_mask_pixel)))
        if sum(u)==0:
                
            for g_id in range(g_bfp.shape[2]):
                g_bfp_1[:,:,g_id]=g_bfp[:,:,g_id]*bfp_phase_exp[:,:]*circ_mask_opt
                    # g_img[:,:,g_id]=1/N*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g_bfp[:,:,g_id])))
                g_img[:,:,g_id]=1/N*np.fft.fftshift(np.fft.fft2((g_bfp_1[:,:,g_id])))
                I_img=I_img+g_img[:,:,g_id]*np.conj(g_img[:,:,g_id])
        else:
                
            for div_pol in range(int(g_bfp.shape[2]/3)):
                for g_id in range(0,3):
                    g_bfp_1[:,:,g_id+div_pol*3]=g_bfp[:,:,g_id+div_pol*3]*bfp_phase_exp[:,:]*circ_mask_opt
                    # g_img[:,:,g_id+div_pol*3]=1/N*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g_bfp[:,:,g_id+div_pol*3])))*u[g_id]
                    g_img[:,:,g_id+div_pol*3]=1/N*np.fft.fftshift(np.fft.fft2((g_bfp_1[:,:,g_id+div_pol*3])))*u[g_id]
                    I_img=I_img+g_img[:,:,g_id]*np.conj(g_img[:,:,g_id])
    else:
        circ_sc=circ_mask
        pupil=  bfp_phase_exp[:,:]*circ_sc
        normfact=1
        pupil=pupil*normfact
            # E_img=1/N*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
        E_img=1/N*np.fft.fftshift(np.fft.fft2((pupil)))
        I_img=E_img*np.conj(E_img)
            
    if d_mask_pixel%2==0:
        if N_crop%2==0:
            I_img_crop=I_img[int(d_mask_pixel/2-N_crop/2):int(d_mask_pixel/2+N_crop/2),int(d_mask_pixel/2-N_crop/2):int(d_mask_pixel/2+N_crop/2)]
        else:
            I_img_crop=I_img[int(d_mask_pixel/2-N_crop/2-0.5):int(d_mask_pixel/2+N_crop/2-0.5),int(d_mask_pixel/2-N_crop/2-0.5):int(d_mask_pixel/2+N_crop/2-0.5)]
    else:
        if N_crop%2==0:
            I_img_crop=I_img[int(d_mask_pixel/2-N_crop/2-0.5):int(d_mask_pixel/2+N_crop/2-0.5),int(d_mask_pixel/2-N_crop/2-0.5):int(d_mask_pixel/2+N_crop/2-0.5)]
        else:
            I_img_crop=I_img[int(d_mask_pixel/2-N_crop/2):int(d_mask_pixel/2+N_crop/2),int(d_mask_pixel/2-N_crop/2):int(d_mask_pixel/2+N_crop/2)]
        
        
    I_img_stack[:,:]=I_img_crop
    return I_img_stack