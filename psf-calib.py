# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:47:12 2022

@author: e0947330
"""

import nd2
import numpy as np
from PIL import Image
from matplotlib.widgets import Cursor
import cv2
import matplotlib.pyplot as plt
import gausscenter
import math
import tifffile


psf = tifffile.imread('G:/tony lab/cx/test.tif')
psf_xy_size = psf.shape[1]
psf_z_size = psf.shape[0]
img=psf[0,:,:]  
imgmax=np.max(np.hstack(img))
img1=255*img/imgmax
img2=img1.astype(np.uint8)
plt.imshow(img2)


def mouse2(event,x,y,flags,param): #Capture the area of interest in the picture
    global img2,point1,point2
    
    img3=img2.copy()
    cv2.imshow('image2',img3)
    if event==cv2.EVENT_LBUTTONDOWN: #Left click
        point1=(x,y)
        cv2.circle(img2,point1,10,(0,255,0),5)
        cv2.imshow('image2',img3)

    elif event==cv2.EVENT_MOUSEMOVE and (flags&cv2.EVENT_FLAG_LBUTTON): #Move the mouse and drag with the left button
        cv2.rectangle(img2,point1,(x,y),(255,0,0),15)
        cv2.imshow('image2',img3)

    elif event==cv2.EVENT_LBUTTONUP: #Left-click to release
        point2=(x,y)
        cv2.rectangle(img2,point1,point2,(0,0,255),5)
        cv2.imshow('image2',img3)
      
cv2.namedWindow('image2')
cv2.setMouseCallback('image2',mouse2) #Capture images 
cv2.imshow('image2',img2)
cv2.waitKey(0)

min_x=min(point1[0],point2[0]) #The two points selected in the picture (top left point and bottom right point)
min_y=min(point1[1],point2[1])
width=abs(point1[0]-point2[0])
height=abs(point1[1]-point2[1])

cut_img=psf[0:psf_z_size,min_y:min_y+height,min_x:min_x+width] #A cropped image collection

cut_img_xsize=cut_img.shape[1]
cut_img_ysize=cut_img.shape[2]

backnoise=np.zeros((psf_z_size))
psf_nobacknoise=np.zeros((psf_z_size,psf_xy_size,psf_xy_size))
for i in range(psf_z_size):
    backnoise[i]=np.sum(cut_img[i,:,:])/(cut_img_xsize*cut_img_ysize)
    psf_nobacknoise[i]=np.rint(psf[i,:,:]-backnoise[i]*np.ones((np.shape(psf[i,:,:]))))
    
psf_nobacknoise[psf_nobacknoise < 0] = 0

intensity = 40000.0
frame_size = 2*psf_z_size-1
pixel=0.064
z_unit=0.1
z_unit_expect=0.05




grid_x, grid_y,grid_z = np.mgrid[0:4.48:141j, 0:4.48:141j,0:2.2:45j]
xc = np.linspace(0,4.48,71)
yc = np.linspace(0,4.48,71)
zc=np.linspace(0,2.2,23)
X, Y ,Z= np.meshgrid(xc, yc,zc)
xc=X[:,:,0].reshape(-1, 1)
yc=Y[:,:,0].reshape(-1, 1)
pointsz=np.zeros((psf_z_size*psf_xy_size*psf_xy_size,1))
pointsx=np.zeros((psf_z_size*psf_xy_size*psf_xy_size,1))
pointsy=np.zeros((psf_z_size*psf_xy_size*psf_xy_size,1))
for i1 in range(psf_z_size):  
    pointsz[i1*5041:(i1+1)*5041]=i1*z_unit
    pointsx[i1*5041:(i1+1)*5041]=xc
    pointsy[i1*5041:(i1+1)*5041]=yc
            
values=np.zeros((psf_z_size*psf_xy_size*psf_xy_size,1))
for i in range((len(pointsz))):
    values[i] = psf[int(pointsz[i]/0.1),int(pointsx[i]/0.064),int(pointsy[i]/0.064)]
points=np.zeros((psf_z_size*psf_xy_size*psf_xy_size,3))
for i in range(115943):
    points[i,0]=pointsx[i]
    points[i,1]=pointsy[i]
    points[i,2]=pointsz[i]

from scipy.interpolate import griddata
grid_v0 = griddata(points, values, (grid_x, grid_y,grid_z), method='nearest')
# grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
# grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
plt.figure()