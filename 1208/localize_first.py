# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:58:51 2022

@author: e0947330
"""

import numpy as _np
import tifffile
import dask.array as _da
import numba as _numba
import multiprocessing as _multiprocessing
import ctypes as _ctypes
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
import threading as _threading
from itertools import chain as _chain
import matplotlib.pyplot as _plt
# import gaussmle as _gaussmle
# import io as _io
import postprocess as _postprocess
# import __main__ as main
import os
from datetime import datetime
import time
from sqlalchemy import create_engine
import pandas as pd
import gausscenter
import matplotlib.patches as patches
import cv2
import math

MAX_LOCS = int(1e6)

_C_FLOAT_POINTER = _ctypes.POINTER(_ctypes.c_float)
LOCS_DTYPE = [
    ("frame", "u4"),
    ("x", "f4"),
    ("y", "f4"),
    ("photons", "f4"),
    ("sx", "f4"),
    ("sy", "f4"),
    ("bg", "f4"),
    ("lpx", "f4"),
    ("lpy", "f4"),
    ("net_gradient", "f4"),
    ("likelihood", "f4"),
    ("iterations", "i4"),
]

MEAN_COLS = [
    "frame",
    "x",
    "y",
    "photons",
    "sx",
    "sy",
    "bg",
    "lpx",
    "lpy",
    "ellipticity",
    "net_gradient",
    "z",
    "d_zcalib",
]
SET_COLS = ["Frames", "Height", "Width", "Box Size", "Min. Net Gradient", "Pixelsize"]

_plt.style.use("ggplot")


@_numba.jit(nopython=True, nogil=True, cache=False)
def local_maxima(frame, box):
    """Finds pixels with maximum value within a region of interest"""
    Y, X = frame.shape
    maxima_map = _np.zeros(frame.shape, _np.uint8)
    box_half = int(box / 2)
    box_half_1 = box_half + 1
    for i in range(box_half, Y - box_half_1):
        for j in range(box_half, X - box_half_1):
            local_frame = frame[
                i - box_half : i + box_half + 1,
                j - box_half : j + box_half + 1,
            ]
            flat_max = _np.argmax(local_frame)
            i_local_max = int(flat_max / box)
            j_local_max = int(flat_max % box)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    y, x = _np.where(maxima_map)
    return y, x


@_numba.jit(nopython=True, nogil=True, cache=False)
def gradient_at(frame, y, x, i):
    gy = frame[y + 1, x] - frame[y - 1, x]
    gx = frame[y, x + 1] - frame[y, x - 1]
    return gy, gx


@_numba.jit(nopython=True, nogil=True, cache=False)
def net_gradient(frame, y, x, box, uy, ux):
    box_half = int(box / 2)
    ng = _np.zeros(len(x), dtype=_np.float32)
    for i, (yi, xi) in enumerate(zip(y, x)):
        for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
            for l_index, m in enumerate(range(xi - box_half, xi + box_half + 1)):
                if not (k == yi and m == xi):
                    gy, gx = gradient_at(frame, k, m, i)
                    ng[i] += gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
    return ng


@_numba.jit(nopython=True, nogil=True, cache=False)
def identify_in_image(image, minimum_ng, box):
    y, x = local_maxima(image, box)
    box_half = int(box / 2)
    # Now comes basically a meshgrid
    ux = _np.zeros((box, box), dtype=_np.float32)
    uy = _np.zeros((box, box), dtype=_np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = _np.sqrt(ux**2 + uy**2)
    ux /= unorm
    uy /= unorm
    ng = net_gradient(image, y, x, box, uy, ux)
    positives = ng > minimum_ng
    y = y[positives]
    x = x[positives]
    ng = ng[positives]
    return y, x, ng


def identify_in_frame(frame, minimum_ng, box, roi=None):
    # print('start identifying in frame')
    if roi is not None:
        frame = frame[roi[0][0] : roi[1][0], roi[0][1] : roi[1][1]]
    image = _np.float32(frame)  # otherwise numba goes crazy
    # print('start identifying in image')
    y, x, net_gradient = identify_in_image(image, minimum_ng, box)
    # print('done identifying in image')
    if roi is not None:
        y += roi[0][0]
        x += roi[0][1]
    # print('done identifying in frame')
    return y, x, net_gradient




cail_psf= _np.load(file="cail_psf.npy")
distance=_np.zeros(cail_psf.shape[0])
angle_cail=_np.zeros(cail_psf.shape[0])
z_position_cail=_np.zeros(cail_psf.shape[0])
for i1 in range(cail_psf.shape[0]):
    frame=cail_psf[i1,:,:]
    coor=gausscenter.gc(frame) #Use Gaussian Fitting
    y=coor[:,0]
    x=coor[:,1] #The coordinates of the spot
    angle_cail[i1]=math.atan((x[0]-x[1])/(y[0]-y[1]))
    # angle_cail[i1]=(x[0]-x[1])/(y[0]-y[1])
    z_position_cail[i1]=i1*0.1
    
    _plt.figure()
    _plt.imshow(frame)
    _plt.scatter(x,y,s=10,c="r")
    distance[i1]=((y[1]-y[0])**2+(x[1]-x[0])**2)**0.5
min_distance=_np.min(distance)
max_distance=_np.max(distance)




Exp_images= _np.array(tifffile.imread('G:/tony lab/cx/E 488 off 561 on -1208,crop.tif'))# read to numpy array
# Exp_images= _np.array(tifffile.imread('G:/tony lab/cx/DH-PSF_TRACKING_SPLINE_MLE/cail.tif'))# read to numpy array
coordinate={}
for i1 in range(Exp_images.shape[0]):
    frame=Exp_images[i1,:,:]
    minimum_ng=1500
    box=7
    roi=None
    y,x,net_gradient=identify_in_frame(frame, minimum_ng, box, roi=None)
    # _plt.figure()
    # _plt.imshow(frame)
    # _plt.scatter(x,y,s=10,c="r")
    coordinate[i1]=(y,x) 

width=30
height=30
total_frame_angle={}
total_frame_Center_coordinate={}
# for i1 in range(Exp_images.shape[0]):
for i1 in range(30,32):
    coordinate_frame=coordinate[i1]
    y_frame= coordinate_frame[0]
    x_frame= coordinate_frame[1]
    j2=0
    Center_coordinate={}
    angle={}
    frame=Exp_images[i1,:,:]
    for xy_index in range(y_frame.shape[0]):
        top_down_y=y_frame[xy_index]-5
        top_up_y=y_frame[xy_index]+5
        top_down_x=x_frame[xy_index]-5
        top_up_x=x_frame[xy_index]+5
        if y_frame[xy_index]<5 :
            top_down_y=0
        if y_frame[xy_index]>Exp_images.shape[1]-5:
            top_up_y=Exp_images.shape[1]
        if x_frame[xy_index]<5 :
            top_down_x=0
        if x_frame[xy_index]>Exp_images.shape[2]-5:
            top_up_x=Exp_images.shape[2]
        location_box=frame[top_down_y:top_up_y+1,top_down_x:top_up_x+1]
        coor1=gausscenter.gc(location_box) #Use Gaussian Fitting
        y_location=coor1[0,0]+top_down_y
        x_location=coor1[0,1]+top_down_x
        y_frame[xy_index]= y_location
        x_frame[xy_index]= x_location

    for i2 in range(y_frame.shape[0]):
        j1=0
        Record={}
        if i2+1<y_frame.shape[0]:
            
            for i3 in range(i2+1,y_frame.shape[0]):
                
                distance_i2_i3=((y_frame[i2]-y_frame[i3])**2+(x_frame[i2]-x_frame[i3])**2)**0.5
                if distance_i2_i3>=min_distance and distance_i2_i3<=max_distance:
                    Record[j1]=[i2,i3]
                    j1=j1+1
           
        if  j1==1:
            Center_coordinate[j2]=[(x_frame[Record[0][0]]+x_frame[Record[0][1]])/2,(y_frame[Record[0][0]]+y_frame[Record[0][1]])/2]
            angle[j2]=math.atan((x_frame[Record[0][0]]-x_frame[Record[0][1]])/(y_frame[Record[0][0]]-y_frame[Record[0][1]]))
            j2=j2+1
            
            #can also find the initial Angle
    top_left_x=_np.zeros(j2)  
    top_left_y=_np.zeros(j2)      
    if not j2==0:
        for j3 in range(j2):
            top_left_x[j3]=Center_coordinate[j3][0]-15
            top_left_y[j3]=Center_coordinate[j3][1]-15
    if not j2==0:     
        max_pixel=_np.max(Exp_images[i1,:,:])
        gray_image=(255*Exp_images[i1,:,:]/max_pixel).astype(_np.uint8)
        for j3 in range(j2):
        # _plt.figure()
        # _plt.imshow(Exp_images[i1,:,:])
        # currentAxis=_plt.gca()
        # rect=patches.Rectangle((top_left_x, top_left_y), width, height, fill=False, edgecolor = 'red',linewidth=1)
        # currentAxis.add_patch(rect)    
            cv2.rectangle(gray_image,(int(top_left_x[j3]), int(top_left_y[j3])), (int(top_left_x[j3]+width), int(top_left_y[j3]+height)), (255, 0, 0), 2) 
    cv2.imshow("gray_img", gray_image)
    cv2.waitKey(0)   
    total_frame_angle[i1]=angle
    total_frame_Center_coordinate[i1]=Center_coordinate







        
plt.figure(8)
plt.imshow(imgSrc)
currentAxis=plt.gca()
rect=patches.Rectangle((200, 600),550,650,linewidth=1,edgecolor='r',facecolor='none')
currentAxis.add_patch(rect)















