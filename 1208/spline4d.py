# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 17:25:21 2022

@author: e0947330
"""

import numpy as  np
from scipy.interpolate import griddata,interp1d
# import matplotlib.pyplot as plt
import cv2
import math


def spline_4D(cail_psf,x_pixel,y_pixel,z_step_size,total_photon):

    input0 =cail_psf
    
    
    input0_size = input0.shape
    input0_x_size=input0_size[2]
    input0_y_size=input0_size[1]
    input0_z_size=input0_size[0]
    
    #remove noise
    # input0_nonoise=np.zeros(input0.shape)
    # for k1 in range(input0_z_size):
    #     mask_image=np.zeros(input0[k1,:,:].shape)
    #     total_energy=np.sum(input0[k1,:,:])
    #     avage_energy=total_energy/(input0_x_size*input0_y_size)
    #     input0_nonoise[k1,:,:]=input0[k1,:,:]-avage_energy*0.2
    #     input0_nonoise[k1,:,:][input0_nonoise[k1,:,:] < 0] = 0
      
    # plt.imshow(input0_nonoise[0,:,:], origin='lower')
    # input0=input0_nonoise
    
    
    
    
    #
    upsample=4
    grid_y, grid_x = np.mgrid[0:(input0_x_size-1)*x_pixel:((upsample*(input0_x_size-1))+input0_x_size)*1j, 0:(input0_y_size-1)*y_pixel:((upsample*(input0_x_size-1))+input0_x_size)*1j]
    
    points =  np.mgrid[0:(input0_x_size-1)*x_pixel:input0_x_size*1j, 0:(input0_y_size-1)*y_pixel:input0_y_size*1j]
    x_points=points[0,:].reshape(-1, 1)
    y_points=points[1,:].reshape(-1, 1)
    xy_points=np.append(x_points,y_points,axis=1)
    spline_xy=np.zeros((((upsample*(input0_x_size-1))+input0_x_size),((upsample*(input0_y_size-1))+input0_y_size),input0_z_size))
    for i in range(input0_z_size):
        
        values_points=input0[i,:].reshape(-1, 1)
        grid_z2 = griddata(xy_points, values_points, (grid_x, grid_y), method='cubic')
        # plt.imshow(grid_z2[:,:,0], origin='lower')
        spline_xy[:,:,i]=grid_z2[:,:,0]
        
    spline_xy_size = spline_xy.shape
    spline_xy_y_size=spline_xy_size[0]
    spline_xy_x_size=spline_xy_size[1]
     
    
    
    
    
    z_points=np.linspace(0,z_step_size*(input0_z_size-1),input0_z_size)
    grid_z=np.linspace(0,z_step_size*(input0_z_size-1),((upsample*(input0_z_size-1))+input0_z_size))
    spline_xyz=np.zeros((((upsample*(input0_x_size-1))+input0_x_size),((upsample*(input0_y_size-1))+input0_y_size),((upsample*(input0_z_size-1))+input0_z_size)))
    for i1 in range(spline_xy_y_size):
        for j1 in range(spline_xy_x_size):
            z_values=spline_xy[i1,j1,:]
            f2 = interp1d(z_points, z_values, kind='cubic')
            spline_xyz[i1,j1,:]=f2(grid_z)
    
    
    # plt.imshow(spline_xyz[:,:,75],origin='lower')
    
    
    matrix_parameter=np.zeros(((input0.shape[0]-1)*(input0.shape[1]-1)*(input0.shape[2]-1),64))
    
    for xyz in range(((input0.shape[0]-1)*(input0.shape[1]-1)*(input0.shape[2]-1))):
        
        tx0=(xyz%(input0_size[2]-1))*x_pixel
        tx0_pixel=(xyz%(input0_size[2]-1))
        tx1=(xyz%(input0_size[2]-1)+1)*x_pixel
        tx1_pixel=(xyz%(input0_size[2]-1)+1)
        ty0=(int(xyz/(input0_size[2]-1))%(input0_size[1]-1))*y_pixel
        ty0_pixel=(int(xyz/(input0_size[2]-1))%(input0_size[1]-1))
        ty1=(int(xyz/(input0_size[2]-1))%(input0_size[1]-1)+1)*y_pixel
        ty1_pixel=(int(xyz/(input0_size[2]-1))%(input0_size[1]-1)+1)
        tz0=(int(xyz/((input0_size[2]-1)*(input0_size[1]-1)))%(input0_size[0]-1))*z_step_size
        tz0_pixel=(int(xyz/((input0_size[2]-1)*(input0_size[1]-1)))%(input0_size[0]-1))
        tz1=(int(xyz/((input0_size[2]-1)*(input0_size[1]-1)))%(input0_size[0]-1)+1)*z_step_size
        tz1_pixel=(int(xyz/((input0_size[2]-1)*(input0_size[1]-1)))%(input0_size[0]-1)+1)
        
        
        
        matrix_xyz=np.zeros((64,64))
        
        
        
        jishu=0
        for mno in range(64):
            
            for m in range(0,4):
                for n in range(0,4):
                    for o in range(0,4):
                        # matrix_grid_xyz[m,n,o]=[(int(jishu/4)%4+1)*(y_pixel/5)+ty0,(jishu%4+1)*(x_pixel/5)+tx0,(int(jishu/16)%4+1)*(z_step_size/5)+tz0]
                        
                        matrix_xyz[mno,m*16+n*4+o]=(((int(jishu/4)%4+1)*(1/5))**n)*(((jishu%4+1)*(1/5))**o)*(((int(jishu/16)%4+1)*(1/5))**m)
                        
            jishu=jishu+1
            
            
        # matrix_points=spline_xyz[tx0_pixel*5+1:tx0_pixel*5+5,ty0_pixel*5+1:ty0_pixel*5+5,tz0_pixel*5+1:tz0_pixel*5+5] 
        matrix_points=spline_xyz[ty0_pixel*5+1:ty0_pixel*5+5,tx0_pixel*5+1:tx0_pixel*5+5,tz0_pixel*5+1:tz0_pixel*5+5]   
        
        
        matrix_points_64=np.zeros((64,1))
        jishu_p=0
        for m in range(0,4):
            for n in range(0,4):
                for o in range(0,4):
                    matrix_points_64[jishu_p,0]=matrix_points[n,o,m]
                    jishu_p=jishu_p+1
                    
        matrix_parameter[xyz,:]=np.dot(np.linalg.inv(matrix_xyz),matrix_points_64).T
    return matrix_parameter              
    #####################################################test    
    # z_test=0
    # pixel_num=64
    # k=int(z_test/z_step_size)
    
    # grid_y_test, grid_x_test = np.mgrid[0:(input0_x_size-1)*x_pixel:1j*pixel_num, 0:(input0_y_size-1)*y_pixel:1j*pixel_num]
    
    # points =  np.mgrid[0:(input0_x_size-1)*x_pixel:input0_x_size*1j, 0:(input0_y_size-1)*y_pixel:input0_y_size*1j]
    
    
    # matrix_num_start=k*(input0.shape[1]-1)*(input0.shape[2]-1)
    
    # img_test=np.zeros((pixel_num,pixel_num))
    
    # for i in range(pixel_num):
    #     for j in range(pixel_num):
    #         # x_test=i*input0_x_size*x_pixel/100
    #         # y_test=j*input0_y_size*y_pixel/100
    #         x_test=grid_x_test[i,j]
    #         y_test=grid_y_test[i,j]
    #         matrix_num=int(x_test/x_pixel)+(int(y_test/y_pixel)*(input0_x_size-1))+matrix_num_start
    #         xyz=matrix_num
    #         if i==pixel_num-1:
    #             i1=pixel_num-2
                
    #             x_test=grid_x_test[i1,j]
    #             y_test=grid_y_test[i1,j]
    #             matrix_num=int(x_test/x_pixel)+(int(y_test/y_pixel)*(input0_x_size-1))+matrix_num_start
    #             xyz=matrix_num
    #         if j==pixel_num-1:
    #             j1=pixel_num-2
                
    #             x_test=grid_x_test[i,j1]
    #             y_test=grid_y_test[i,j1]
    #             matrix_num=int(x_test/x_pixel)+(int(y_test/y_pixel)*(input0_x_size-1))+matrix_num_start
    #             xyz=matrix_num
            
    #         if j==pixel_num-1 and i==pixel_num-1:
    #             j1=pixel_num-2
    #             i1=pixel_num-2
    #             x_test=grid_x_test[i1,j1]
    #             y_test=grid_y_test[i1,j1]
    #             matrix_num=int(x_test/x_pixel)+(int(y_test/y_pixel)*(input0_x_size-1))+matrix_num_start
    #             xyz=matrix_num
                
                
    #         tx0=(xyz%(input0_size[2]-1))*x_pixel
    #         tx0_pixel=(xyz%(input0_size[2]-1))
    #         tx1=(xyz%(input0_size[2]-1)+1)*x_pixel
    #         tx1_pixel=(xyz%(input0_size[2]-1)+1)
    #         ty0=(int(xyz/(input0_size[2]-1))%(input0_size[1]-1))*y_pixel
    #         ty0_pixel=(int(xyz/(input0_size[2]-1))%(input0_size[1]-1))
    #         ty1=(int(xyz/(input0_size[2]-1))%(input0_size[1]-1)+1)*y_pixel
    #         ty1_pixel=(int(xyz/(input0_size[2]-1))%(input0_size[1]-1)+1)
    #         tz0=(int(xyz/((input0_size[2]-1)*(input0_size[1]-1)))%(input0_size[0]-1))*z_step_size
    #         tz0_pixel=(int(xyz/((input0_size[2]-1)*(input0_size[1]-1)))%(input0_size[0]-1))
    #         tz1=(int(xyz/((input0_size[2]-1)*(input0_size[1]-1)))%(input0_size[0]-1)+1)*z_step_size
    #         tz1_pixel=(int(xyz/((input0_size[2]-1)*(input0_size[1]-1)))%(input0_size[0]-1)+1)
    #         for m in range(0,4):
    #             for n in range(0,4):
    #                 for o in range(0,4):
                        
    #                     img_test[i,j]=matrix_parameter[matrix_num,m*16+n*4+o]*(((x_test-tx0)/(tx1-tx0))**o)*(((y_test-ty0)/(ty1-ty0))**n)*(((z_test-tz0)/(tz1-tz0))**m)+img_test[i,j]
            
    
    # plt.imshow(img_test,origin='lower')
    # img_test1 = cv2.resize(img_test,(32,32), interpolation=cv2.INTER_CUBIC)
            
    # erro=np.abs(img_test1-input0_nonoise[0,:,:])/input0_nonoise[17,:,:]
    # plt.imshow(erro,origin='lower')