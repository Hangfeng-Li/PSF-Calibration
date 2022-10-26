# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 22:25:53 2022

@author: lihsn
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat

#读取
# filename = 'C:/Users/lihsn/Desktop/MBI/Tony Lab/cx/cab/mask.mat'
# mask0 = loadmat(filename)
# print(type(mask0))
# mask = mask0['mask1']
# plt.imshow(mask, origin='lower')

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
x_position=0
y_position=0
z_position=z_emit
NFP=np.linspace(-2,2,11)
FOV_size  =70



d_mask=lamda*f_4f/(pixel_size_CCD*pixel_size_SLM)*pixel_size_SLM
d_mask_pixel=np.ceil(lamda*f_4f/(pixel_size_CCD*pixel_size_SLM))
d_bfp=2*f_4f*NA/math.sqrt(np.square(M)-np.square(NA))
d_bfp_pixel=np.ceil(d_bfp/(pixel_size_SLM))


mask=np.zeros((int(d_mask_pixel),int(d_mask_pixel)))



x_or_y_phys=pixel_size_SLM*(np.arange(1,int((d_mask_pixel)+1))-np.ceil(d_mask_pixel/2))

phys_y_SLM,phys_x_SLM=np.mgrid[x_or_y_phys[0]:x_or_y_phys[list(x_or_y_phys.shape)[0]-1]:1j*list(x_or_y_phys.shape)[0],x_or_y_phys[0]:x_or_y_phys[list(x_or_y_phys.shape)[0]-1]:1j*list(x_or_y_phys.shape)[0]]

x_normal=np.linspace(-1,1,int(d_mask_pixel))*d_mask_pixel/(d_bfp/pixel_size_SLM)

phys_y_normal_SLM,phys_x_normal_SLM=np.mgrid[x_normal[0]:x_normal[int(d_mask_pixel)-1]:1j*int(d_mask_pixel),x_normal[0]:x_normal[int(d_mask_pixel)-1]:1j*int(d_mask_pixel)]

r_normal_SLM=np.sqrt(np.power(phys_y_normal_SLM,2)+np.power(phys_x_normal_SLM,2))
mask_NA=(r_normal_SLM<=1)

sin_theta_glass=NA/n_glass*r_normal_SLM
cos_theta_glass=(1-np.power(sin_theta_glass,2)+0*1j)**0.5

sin_theta_medium=n_glass/n_medium*sin_theta_glass
cos_theta_medium=(1-np.power(sin_theta_medium,2)+0*1j)**0.5

circ_mask_medium= ~(abs(np.imag(cos_theta_medium))>0)

circ_mask=mask_NA*circ_mask_medium
circ_mask_large=mask_NA


if SAF_flag==0:
    tmp=circ_mask
else:
    tmp=circ_mask_large


# plt.imshow(circ_mask, origin='lower')
# plt.imshow(circ_mask_large, origin='lower')

int_amp_correction=1/((cos_theta_glass+0*1j)**0.5)


phi=np.zeros((int(d_mask_pixel),int(d_mask_pixel),4))
phi[:,:,0]=phys_x_SLM*M/f_4f#x shift phase
phi[:,:,1]=phys_y_SLM*M/f_4f#y shift phase
phi[:,:,2]=n_medium*cos_theta_medium#depth phase
phi[:,:,3]=n_glass*cos_theta_glass# defocus phase

#Fresnel transmission coefficient tp ts:
tp=(2*n_medium*cos_theta_medium)/(n_glass*cos_theta_medium+n_medium*cos_theta_glass)
ts=(2*n_medium*cos_theta_medium)/(n_medium*cos_theta_medium+n_glass*cos_theta_glass)

#transfer coeficient:
c1=((n_glass/n_medium)**2)*(cos_theta_glass/cos_theta_medium)*tp
c2=(n_glass/n_medium)*tp
c3=(n_glass/n_medium)*(cos_theta_glass/cos_theta_medium)*ts

c1[np.isnan(c1)] = 0
c2[np.isnan(c2)] = 0
c3[np.isnan(c3)] = 0

#BFP angular coordinate
sin_phi=phys_y_normal_SLM/((phys_x_normal_SLM**2+phys_y_normal_SLM**2)**0.5)
cos_phi=phys_x_normal_SLM/((phys_x_normal_SLM**2+phys_y_normal_SLM**2)**0.5)

if sin_phi.shape[0]%2==1:
    sin_phi[int(np.ceil(sin_phi.shape[0]/2))-1,int(np.ceil(sin_phi.shape[0]/2))-1]=1
    cos_phi[int(np.ceil(sin_phi.shape[0]/2))-1,int(np.ceil(sin_phi.shape[0]/2))-1]=1

#Green matrix bfp
g_bfp_xx=c3*(sin_phi**2)+c2*(cos_phi**2)*cos_theta_glass
g_bfp_xy=-sin_phi*cos_phi*(c3-c2*cos_theta_glass)
g_bfp_xz=-c1*cos_phi*sin_theta_glass
g_bfp_yx=-sin_phi*cos_phi*(c3-c2*cos_theta_glass)
g_bfp_yy=c3*(cos_phi**2)+c2*(sin_phi**2)*cos_theta_glass
g_bfp_yz=-c1*sin_phi*sin_theta_glass

#SAF decay 
bfp_decay=np.abs(np.exp(1j*k*z_emit*phi[:,:,2]))*tmp




if polar_flag==0:
    g_bfp=np.zeros((int(d_mask_pixel),int(d_mask_pixel),3))
    g_bfp[:,:,0]=int_amp_correction*g_bfp_xx*tmp
    g_bfp[:,:,1]=int_amp_correction*g_bfp_xy*tmp
    g_bfp[:,:,2]=int_amp_correction*g_bfp_xz*tmp
    
elif polar_flag==1:
    g_bfp=np.zeros((int(d_mask_pixel),int(d_mask_pixel),3))
    g_bfp[:,:,0]=int_amp_correction*g_bfp_yx*tmp
    g_bfp[:,:,1]=int_amp_correction*g_bfp_yy*tmp
    g_bfp[:,:,2]=int_amp_correction*g_bfp_yz*tmp
    
else:
    g_bfp=np.zeros((int(d_mask_pixel),int(d_mask_pixel),6))
    g_bfp[:,:,0]=int_amp_correction*g_bfp_xx*tmp
    g_bfp[:,:,1]=int_amp_correction*g_bfp_xy*tmp
    g_bfp[:,:,2]=int_amp_correction*g_bfp_xz*tmp
    g_bfp[:,:,3]=int_amp_correction*g_bfp_yx*tmp
    g_bfp[:,:,4]=int_amp_correction*g_bfp_yy*tmp
    g_bfp[:,:,5]=int_amp_correction*g_bfp_yz*tmp
    
if np.sum(u)==0:#free rotating u=[0,0,0]
    sum_bfp=np.zeros(g_bfp.shape[2])
    for g_id in range(g_bfp.shape[2]):
        sum_bfp[g_id]=np.sum(np.sum(np.abs(g_bfp[:,:,g_id]*bfp_decay)))
    normfact=np.sqrt(signal)/np.sqrt(np.sum(sum_bfp))
else:
    p_bfp=np.zeros((int(d_mask_pixel),int(d_mask_pixel),3))
    sum_bfp=np.zeros(int(g_bfp.shape[2]/3))
    for div_pol in range(int(g_bfp.shape[2]/3)):
        for g_id in range(0,3):
            p_bfp[:,:,g_id]=g_bfp[:,:,g_id+div_pol*3]*bfp_decay*u[g_id]
        
        sum_bfp[div_pol]=np.sum(np.sum(np.abs(np.sum(p_bfp,2)**2)))
        
normfact=np.sqrt(signal)/np.sqrt(np.sum(sum_bfp))
    
circ_mask_opt=tmp
    
N=tmp.shape[0]
N_crop=FOV_size   

bfp_phase=np.zeros((int(d_mask_pixel),int(d_mask_pixel),NFP.shape[0]))
bfp_phase_exp=np.zeros((int(d_mask_pixel),int(d_mask_pixel),NFP.shape[0]))+0*1j
for NFP_id in range(NFP.shape[0]):
    
    bfp_phase[:,:,NFP_id]=k*(z_position*phi[:,:,2]+x_position*phi[:,:,0]+y_position*phi[:,:,1]+NFP[NFP_id]*phi[:,:,3])  
    bfp_phase[:,:,NFP_id][np.isnan(bfp_phase[:,:,NFP_id])]=0
    bfp_phase_exp[:,:,NFP_id]=np.exp(1j*(bfp_phase[:,:,NFP_id]+mask))

I_img_stack=np.zeros((N_crop,N_crop,NFP.shape[0]))   
                     
for NFP_id in range(NFP.shape[0]):
    
    if vec_flag==1:
        g_img=np.zeros((int(d_mask_pixel),int(d_mask_pixel),g_bfp.shape[2]))+0*1j
        g_bfp_1=np.zeros((int(d_mask_pixel),int(d_mask_pixel),g_bfp.shape[2]))+0*1j
        I_img=np.zeros((int(d_mask_pixel),int(d_mask_pixel)))
        if sum(u)==0:
            
            for g_id in range(g_bfp.shape[2]):
                g_bfp_1[:,:,g_id]=g_bfp[:,:,g_id]*bfp_phase_exp[:,:,NFP_id]*circ_mask_opt
                # g_img[:,:,g_id]=1/N*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g_bfp[:,:,g_id])))
                g_img[:,:,g_id]=1/N*np.fft.fftshift(np.fft.fft2((g_bfp_1[:,:,g_id])))
                I_img=I_img+g_img[:,:,g_id]*np.conj(g_img[:,:,g_id])
        else:
            
            for div_pol in range(int(g_bfp.shape[2]/3)):
                for g_id in range(0,3):
                    g_bfp_1[:,:,g_id+div_pol*3]=g_bfp[:,:,g_id+div_pol*3]*bfp_phase_exp[:,:,NFP_id]*circ_mask_opt
                    # g_img[:,:,g_id+div_pol*3]=1/N*np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g_bfp[:,:,g_id+div_pol*3])))*u[g_id]
                    g_img[:,:,g_id+div_pol*3]=1/N*np.fft.fftshift(np.fft.fft2((g_bfp_1[:,:,g_id+div_pol*3])))*u[g_id]
                    I_img=I_img+g_img[:,:,g_id]*np.conj(g_img[:,:,g_id])
    else:
        circ_sc=circ_mask
        pupil=  bfp_phase_exp[:,:,NFP_id]*circ_sc
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
    
    
    I_img_stack[:,:,NFP_id]=I_img_crop

plt.imshow(I_img_stack[:,:,5], origin='lower')  
# plt.imshow(circ_mask_opt , origin='lower')



  
    
    






