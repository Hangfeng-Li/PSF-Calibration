# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:49:18 2022

@author: e0947330
"""
import tifffile
import numpy as  np
from scipy.interpolate import griddata,interp1d
import matplotlib.pyplot as plt
import cv2




input0 = np.array(tifffile.imread('G:/tony lab/cx/cailb4.tif'),dtype=float)
x_pixel=0.065
y_pixel=0.065#um
z_step_size=0.1
total_photon=4000

input0_size = input0.shape
input0_x_size=input0_size[2]
input0_y_size=input0_size[1]
input0_z_size=input0_size[0]

#remove noise
input0_nonoise=np.zeros(input0.shape)
for k1 in range(input0_z_size):
    mask_image=np.zeros(input0[k1,:,:].shape)
    total_energy=np.sum(input0[k1,:,:])
    avage_energy=total_energy/(input0_x_size*input0_y_size)
    input0_nonoise[k1,:,:]=input0[k1,:,:]-avage_energy*0.2
    input0_nonoise[k1,:,:][input0_nonoise[k1,:,:] < 0] = 0
    # imgmax=np.max(np.hstack(input0[k1,:,:]))
    # mask_image[input0[k1,:,:] < 0.2*imgmax] = 1
    
    # img1=255*input0[k1,:,:]/imgmax
    # orign_image_uint8=img1.astype(np.uint8)
    # gray_img = orign_image_uint8
    # ret,mask_image=cv2.threshold(gray_img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # backnoise=(np.sum(input0[k1,:,:])-np.sum((mask_image*input0[k1,:,:])))/np.sum(mask_image)
    # psf_nobacknoise=np.rint(input0[k1,:,:]-backnoise*np.ones((np.shape(input0[k1,:,:]))))
    # psf_nobacknoise[psf_nobacknoise < 0] = 0
    # input0_nonoise[k1,:,:]=psf_nobacknoise
plt.imshow(input0_nonoise[1,:,:], origin='lower')
input0=input0_nonoise

mask_extract0=np.zeros((input0_x_size,input0_y_size))
for k1 in range(input0_z_size):
    mask_extract0=input0_nonoise[k1,:,:]+mask_extract0
    
mask_extract1=100*mask_extract0/np.sum(mask_extract0)
mask_extract2=mask_extract1-np.max(mask_extract1)*0.8
mask_extract2[mask_extract2<0]=0
plt.imshow(mask_extract2, origin='lower')

#xy
# average_band_y=int(0.4*input0_y_size)
# average_band_x=int(0.4*input0_x_size)
# if average_band_y%2==1:
#     average_band_y=average_band_y+1
# if average_band_x%2==1:
#     average_band_x=average_band_x+1
# for k in range(input0_z_size):
#     P_x=np.zeros(input0_x_size)
#     P_y=np.zeros(input0_y_size)
#     P_x_n=np.zeros(input0_x_size)
#     P_y_n=np.zeros(input0_y_size)
#     if input0_x_size%2==0:
#         for j1 in range(int(((input0_x_size/2)-1)-(average_band_x/2)),int(((input0_x_size/2))+(average_band_x/2))):
#            P_y= input0[k,:,j1]/average_band_y+P_y
#     else:
#         for j1 in range(int(((input0_x_size-1)/2)-(average_band_x/2)),int(((input0_x_size-1)/2)+(average_band_x/2))):
#            P_y= input0[k,:,j1]/average_band_y+P_y
#     if input0_y_size%2==0:
#         for i1 in range(int(((input0_y_size/2)-1)-(average_band_y/2)),int(((input0_y_size/2))+(average_band_y/2))):
#            P_x= input0[k,i1,:]/average_band_x+P_x
#     else:
#         for i1 in range(int(((input0_y_size-1)/2)-(average_band_y/2)),int(((input0_y_size-1)/2)+(average_band_y/2))):
#            P_x= input0[k,i1,:]/average_band_x+P_x   
#     for i1 in range(input0_x_size):
#         P_x_n[i1]=P_x[-i1]
#     for j1 in range(input0_y_size):
#         P_y_n[j1]=P_y[-j1]
#     x_corr=np.correlate(P_x,P_x_n,'same')
#     y_corr=np.correlate(P_y,P_y_n,'same')
    
# xx_points=np.linspace(0,x_pixel*input0_x_size,input0_x_size)
# yy_points=np.linspace(0,y_pixel*input0_y_size,input0_y_size)    
# plt.scatter(xx_points, x_corr)
# plt.show()  
# plt.scatter(yy_points, y_corr)
# plt.show()  
  
#
upsample=4
grid_x, grid_y = np.mgrid[0:input0_x_size*x_pixel:((upsample*input0_x_size)-(upsample-1))*1j, 0:input0_y_size*y_pixel:((upsample*input0_y_size)-(upsample-1))*1j]

points =  np.mgrid[0:input0_x_size*x_pixel:input0_x_size*1j, 0:input0_y_size*y_pixel:input0_y_size*1j]
x_points=points[0,:].reshape(-1, 1)
y_points=points[1,:].reshape(-1, 1)
xy_points=np.append(x_points,y_points,axis=1)
spline_xy=np.zeros(((upsample*input0_x_size)-(upsample-1),(upsample*input0_y_size)-(upsample-1),input0_z_size))
for i in range(input0_z_size):
    
    values_points=input0[i,:].reshape(-1, 1)
    grid_z2 = griddata(xy_points, values_points, (grid_x, grid_y), method='cubic')
    # plt.imshow(grid_z2[:,:,0], origin='lower')
    spline_xy[:,:,i]=grid_z2[:,:,0]
    
spline_xy_size = spline_xy.shape
spline_xy_y_size=spline_xy_size[0]
spline_xy_x_size=spline_xy_size[1]
 




z_points=np.linspace(0,z_step_size*(input0_z_size-1),input0_z_size)
grid_z=np.linspace(0,z_step_size*(input0_z_size-1),((upsample*input0_z_size)-(upsample-1)))
spline_xyz=np.zeros(((upsample*input0_x_size)-(upsample-1),(upsample*input0_y_size)-(upsample-1),(upsample*input0_z_size)-(upsample-1)))
for i1 in range(spline_xy_y_size):
    for j1 in range(spline_xy_x_size):
        z_values=spline_xy[i1,j1,:]
        f2 = interp1d(z_points, z_values, kind='cubic')
        spline_xyz[i1,j1,:]=f2(grid_z)
        

# plt.imshow(spline_xyz[:,:,36], origin='lower')
# plt.imshow(input0[9,:,:], origin='lower')
fft2_spline_xyz=np.zeros(spline_xyz.shape,dtype=complex)
spline_xyz_photon=np.zeros(spline_xyz.shape,dtype=complex)
for k1 in range((upsample*input0_z_size)-(upsample-1)):
    spline_xyz_photon[:,:,k1]=total_photon*spline_xyz[:,:,k1]/np.sum(spline_xyz[:,:,k1])
    # fft2_spline_xyz[:,:,k1]=np.fft.fftshift(np.fft.fft2(spline_xyz[:,:,k1]))
    fft2_spline_xyz[:,:,k1]=np.fft.fft2(spline_xyz[:,:,k1])
        

