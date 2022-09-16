# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:07:17 2022

@author: lihsn
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:23:53 2022

@author: lihsn
"""

import nd2
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
global img
global point1,point2


input0 = nd2.imread('C:/Users/lihsn/Desktop/MBI/Tony Lab/cx/E 488 off 561 on 001.nd2') # read to numpy array
size = input0.shape
# input1=input0[0,:,:]
# plt.imshow(input1)
# input1max=np.max(np.hstack(input1))
# input3=255*input1/input1max
# input4=input3.astype(np.uint8)
# image_array=input3
# #image_array *= 255  # 变换为0-255的灰度值
# img0 = Image.fromarray(image_array)
# img1 = img0.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
# plt.imshow(img1)
# img=np.array(img1)
input6=np.ones(input0.shape)

for i in range(size[0]): #Input image to Uint8 format,each image was normalized to its highest intensity
    input1max=np.max(np.hstack(input0[i,:,:]))
    input5=255*input0[i,:,:]/input1max
    input6[i,:,:]=input5.astype(np.uint8)
input6=input6.astype(np.uint8)

def mouse1(event,x,y,flags,param): #Use the mouse wheel to browse the picture stack, and use the middle click to achieve the extraction of the image sequence number of interest
    global zoom,runimage,flag1,RECORDZOOM
    text="NUM:"+str(zoom)
    runimage=input6[zoom,:,:]     
    cv2.putText(runimage,text,(100,100),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2)
    cv2.imshow('image',runimage)
    if event == cv2.EVENT_MOUSEWHEEL:  # Mouse wheel event
       
       if  zoom==0:
             if flags > 0:  # Move the mouse wheel up
                 zoom += wheel_step
                 # runimage=input6[zoom,:,:]
             else:  # The mouse wheel moves down
                 zoom -= 0   
       elif  zoom==(size[0]-1):
              if flags < 0:  # 
                  zoom -= wheel_step
                  # runimage=input6[zoom,:,:]
              else:  # 
                  zoom -= 0   
       else:
                       if flags > 0:  # Mouse wheel event
                           zoom += wheel_step
                           # runimage=input6[zoom,:,:]
                       else:  # The mouse wheel moves down
                           zoom -= wheel_step
    elif event == cv2.EVENT_MBUTTONDOWN: #In total, you can click the middle key twice, and finally select the images in the middle sequence of the two clicks for clipping.
        RECORDZOOM[flag1]=zoom
        flag1 += 1
        
    else:
        zoom -= 0
   

     
wheel_step, zoom = 1, 0  # Mouse wheel step value and initial value
flag1=0
RECORDZOOM=[0,1] #Stores the selected sequence numbers
runimage=input6[0,:,:]
cv2.namedWindow('image')
cv2.setMouseCallback('image',mouse1) #Browse the picture and select the picture
cv2.waitKey(0)

lengthimg=RECORDZOOM[1]-RECORDZOOM[0] #The size of the selected image sequence

def mouse2(event,x,y,flags,param): #Capture the area of interest in the picture
    global img,point1,point2
    img=input6[RECORDZOOM[0],:,:]  
    img2=img.copy()
    cv2.imshow('image2',img2)
    if event==cv2.EVENT_LBUTTONDOWN: #Left click
        point1=(x,y)
        cv2.circle(img2,point1,10,(0,255,0),5)
        cv2.imshow('image2',img2)

    elif event==cv2.EVENT_MOUSEMOVE and (flags&cv2.EVENT_FLAG_LBUTTON): #Move the mouse and drag with the left button
        cv2.rectangle(img2,point1,(x,y),(255,0,0),15)
        cv2.imshow('image2',img2)

    elif event==cv2.EVENT_LBUTTONUP: #Left-click to release
        point2=(x,y)
        cv2.rectangle(img2,point1,point2,(0,0,255),5)
        cv2.imshow('image2',img2)
      




cv2.namedWindow('image2')
cv2.setMouseCallback('image2',mouse2) #Capture images 
cv2.imshow('image2',img)
cv2.waitKey(0)

min_x=min(point1[0],point2[0]) #The two points selected in the picture (top left point and bottom right point)
min_y=min(point1[1],point2[1])
width=abs(point1[0]-point2[0])
height=abs(point1[1]-point2[1])

cut_img=input0[RECORDZOOM[0]:RECORDZOOM[1],min_y:min_y+height,min_x:min_x+width] #A cropped image collection



