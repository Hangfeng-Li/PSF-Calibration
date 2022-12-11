# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:52:14 2022

@author: e0947330
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:22:10 2022

@author: e0947330
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:57:46 2022

@author: e0947330
"""
import numpy as  np
import cv2
import numba 

# @numba.jit(nopython=True)
def dev_cail_psf(z_test0,y_test0,x_test0,h_test0,b_test0,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter):
   
    
    # x_test0=0
    # y_test0=0
    # z_test0=0.4
    # h_test0=90
    # b_test0=3
    
    # input0_x_size=33
    # input0_y_size=33
    # input0_z_size=22
    
    # z_step_size=0.1
    # x_pixel=0.065
    # y_pixel=0.065
    # cail_psf_size=cail_psf.shape
    
    
    # big_win_size_small=73
    # win_size_small=33
    
   
    
   
    
   
    
    
    pixel_num=big_win_size_small*2
    k=int(z_test0/z_step_size)
     
    grid_y_test, grid_x_test = np.mgrid[0:(big_win_size_small-1)*x_pixel:1j*pixel_num, 0:(big_win_size_small-1)*y_pixel:1j*pixel_num]
    grid_y_test=grid_y_test-y_test0
    grid_x_test=grid_x_test-x_test0
    
    
     
    matrix_num_start=k*(big_win_size_small-1)*(big_win_size_small-1)
     
    dev_h_img_test=np.zeros((pixel_num,pixel_num))
    dev_x_img_test= np.zeros((pixel_num,pixel_num))
    dev_y_img_test= np.zeros((pixel_num,pixel_num))
    dev_z_img_test= np.zeros((pixel_num,pixel_num))
    
    for i in range(pixel_num):
        for j in range(pixel_num):
            # x_test=i*input0_x_size*x_pixel/100
            # y_test=j*input0_y_size*y_pixel/100
            x_test=grid_x_test[i,j]
            y_test=grid_y_test[i,j]
            x_test1=x_test
            y_test1=y_test
            # if x_test>=(win_size_small-1)*x_pixel:
            #     x_test1=(win_size_small-1)*x_pixel-(grid_x_test[0,1]-grid_x_test[0,0])
            # if x_test<0:
            #     x_test1=0
            # if y_test>=(win_size_small-1)*y_pixel:
            #     y_test1=(win_size_small-1)*y_pixel-(grid_y_test[1,0]-grid_y_test[0,0])
            # if y_test<0:
            #     y_test1=0
            
            # if x_test>=(win_size_small-1)*x_pixel:
            #     x_test1=(win_size_small-1)*x_pixel-(grid_x_test[0,1]-grid_x_test[0,0])
            #     x_test=x_test1
            # if x_test<0:
            #     x_test1=0
            #     x_test=x_test1
            # if y_test>=(win_size_small-1)*y_pixel:
            #     y_test1=(win_size_small-1)*y_pixel-(grid_y_test[1,0]-grid_y_test[0,0])
            #     y_test=y_test1
            # if y_test<0:
            #     y_test1=0
            #     y_test=y_test1
            if x_test<(big_win_size_small-1)*x_pixel and x_test>=0 and y_test<(big_win_size_small-1)*y_pixel and y_test>=0:
            
                matrix_num=int(x_test1/x_pixel)+(int(y_test1/y_pixel)*(cail_psf_size[2]-1))+matrix_num_start
                xyz=matrix_num
                
                tx0=(xyz%(cail_psf_size[2]-1))*x_pixel
                tx0_pixel=(xyz%(cail_psf_size[2]-1))
                tx1=(xyz%(cail_psf_size[2]-1)+1)*x_pixel
                tx1_pixel=(xyz%(cail_psf_size[2]-1)+1)
                ty0=(int(xyz/(cail_psf_size[2]-1))%(cail_psf_size[1]-1))*y_pixel
                ty0_pixel=(int(xyz/(cail_psf_size[2]-1))%(cail_psf_size[1]-1))
                ty1=(int(xyz/(cail_psf_size[2]-1))%(cail_psf_size[1]-1)+1)*y_pixel
                ty1_pixel=(int(xyz/(cail_psf_size[2]-1))%(cail_psf_size[1]-1)+1)
                tz0=(int(xyz/((cail_psf_size[2]-1)*(cail_psf_size[1]-1)))%(cail_psf_size[0]-1))*z_step_size
                tz0_pixel=(int(xyz/((cail_psf_size[2]-1)*(cail_psf_size[1]-1)))%(cail_psf_size[0]-1))
                tz1=(int(xyz/((cail_psf_size[2]-1)*(cail_psf_size[1]-1)))%(cail_psf_size[0]-1)+1)*z_step_size
                tz1_pixel=(int(xyz/((cail_psf_size[2]-1)*(cail_psf_size[1]-1)))%(cail_psf_size[0]-1)+1)
                for m in range(0,4):
                    for n in range(0,4):
                        for o in range(0,4):
                           
                            dev_h_img_test[i,j]=matrix_parameter[matrix_num,m*16+n*4+o]*(((x_test-tx0)/(tx1-tx0))**o)*(((y_test-ty0)/(ty1-ty0))**n)*(((z_test0-tz0)/(tz1-tz0))**m)+dev_h_img_test[i,j]
                            dev_x_img_test[i,j]=((o+1)/(tx1-tx0))*matrix_parameter[matrix_num,m*16+n*4+o]*(((x_test-tx0)/(tx1-tx0))**o)*(((y_test-ty0)/(ty1-ty0))**n)*(((z_test0-tz0)/(tz1-tz0))**m)+dev_x_img_test[i,j]
                            dev_y_img_test[i,j]=((n+1)/(ty1-ty0))*matrix_parameter[matrix_num,m*16+n*4+o]*(((x_test-tx0)/(tx1-tx0))**o)*(((y_test-ty0)/(ty1-ty0))**n)*(((z_test0-tz0)/(tz1-tz0))**m)+dev_y_img_test[i,j]
                            dev_z_img_test[i,j]=((m+1)/(tz1-tz0))*matrix_parameter[matrix_num,m*16+n*4+o]*(((x_test-tx0)/(tx1-tx0))**o)*(((y_test-ty0)/(ty1-ty0))**n)*(((z_test0-tz0)/(tz1-tz0))**m)+dev_z_img_test[i,j]
    
    
    dev_h_img_test1 = cv2.resize(dev_h_img_test,(big_win_size_small,big_win_size_small), interpolation=cv2.INTER_CUBIC)
    dev_x_img_test1 = cv2.resize(dev_x_img_test,(big_win_size_small,big_win_size_small), interpolation=cv2.INTER_CUBIC)
    dev_y_img_test1 = cv2.resize(dev_y_img_test,(big_win_size_small,big_win_size_small), interpolation=cv2.INTER_CUBIC)
    dev_z_img_test1 = cv2.resize(dev_z_img_test,(big_win_size_small,big_win_size_small), interpolation=cv2.INTER_CUBIC)
    
    
    
    
    dev_x_img_test1=-1*h_test0*dev_x_img_test1
    dev_y_img_test1=-1*h_test0*dev_y_img_test1
    dev_z_img_test1=-1*h_test0*dev_z_img_test1
    
    dev_h_img_test2=dev_h_img_test1[20:53,20:53]
    dev_x_img_test2=dev_x_img_test1[20:53,20:53]
    dev_y_img_test2=dev_y_img_test1[20:53,20:53]
    dev_z_img_test2=dev_z_img_test1[20:53,20:53]
    return dev_h_img_test2,dev_x_img_test2,dev_y_img_test2,dev_z_img_test2





































