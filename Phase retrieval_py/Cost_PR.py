# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:11:00 2022

@author: e0947330
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import cv2


def cost_PR(N,mask,NFP0,x_position0,y_position0,z_position0,data_stack,N_bg_stack,std_bg_stack,phi,k,d_mask_pixel,vec_flag,g_bfp,u,circ_mask_opt,circ_mask,N_crop,gBlur,cost_function_flag):
    
    grad=np.zeros(mask.shape)
    I_img=np.zeros(mask.shape)
    cost=np.zeros(NFP0.shape[0])
    
    
    for z_ind in range(NFP0.shape[0]):
        I_img=np.zeros(mask.shape)
        x0=x_position0[z_ind]
        y0=y_position0[z_ind]
        z0=z_position0[z_ind]
        NFP=NFP0[z_ind]
        
        data_NFP=data_stack[:,:,z_ind]
        N_bg=N_bg_stack[z_ind]
        std_bg=std_bg_stack[z_ind]
        
        
        
        bfp_phase=np.zeros((int(d_mask_pixel),int(d_mask_pixel)))
        bfp_phase_exp=np.zeros((int(d_mask_pixel),int(d_mask_pixel)))+0*1j
        bfp_phase[:,:]=k*(z0*phi[:,:,2]+x0*phi[:,:,0]+y0*phi[:,:,1]-NFP*phi[:,:,3])  
        bfp_phase[:,:][np.isnan(bfp_phase[:,:])]=0
        bfp_phase_exp[:,:]=np.exp(1j*(bfp_phase[:,:]+mask))
        
        if vec_flag==1:
            g_img=np.zeros((int(d_mask_pixel),int(d_mask_pixel),g_bfp.shape[2]))+0*1j
            g_bfp_1=np.zeros((int(d_mask_pixel),int(d_mask_pixel),g_bfp.shape[2]))+0*1j
            I_img=np.zeros((int(d_mask_pixel),int(d_mask_pixel)))
            if sum(abs(u)==0)==3:
                    
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
            
        normfact=N_bg/sum(I_img)
        I_img=I_img*normfact
        tmp=cv2.filter2D(I_img, gBlur)
        ####
        ###
        ##
        #
        
        if cost_function_flag==1:
            cost[z_ind]=sum(abs(tmp-data_NFP))
            dcost_dtmp=np.sign(tmp-data_NFP)*data_NFP
        
        
        
        
        