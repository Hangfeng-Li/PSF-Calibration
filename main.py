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
input1=input0[0,:,:]
plt.imshow(input1)
input1max=np.max(np.hstack(input1))
input3=255*input1/input1max
input4=input3.astype(np.uint8)
image_array=input3
#image_array *= 255  # 变换为0-255的灰度值
img0 = Image.fromarray(image_array)
img1 = img0.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
plt.imshow(img1)
img=np.array(img1)

def on_mouse(event,x,y,flags,param):
    global img,point1,point2
    img2=img.copy()
    if event==cv2.EVENT_LBUTTONDOWN:#左键点击
        point1=(x,y)
        cv2.circle(img2,point1,10,(0,255,0),5)
        cv2.imshow('image',img2)

    elif event==cv2.EVENT_MOUSEMOVE and (flags&cv2.EVENT_FLAG_LBUTTON):#移动鼠标，左键拖拽
        cv2.rectangle(img2,point1,(x,y),(255,0,0),15)#需要确定的就是矩形的两个点（左上角与右下角），颜色红色，线的类型（不设置就默认）。
        cv2.imshow('image',img2)

    elif event==cv2.EVENT_LBUTTONUP:#左键释放
        point2=(x,y)
        cv2.rectangle(img2,point1,point2,(0,0,255),5)#需要确定的就是矩形的两个点（左上角与右下角），颜色蓝色，线的类型（不设置就默认）。
        cv2.imshow('image',img2)
      

cv2.namedWindow('image')
cv2.setMouseCallback('image',on_mouse)
cv2.imshow('image',img)
cv2.waitKey(0)

min_x=min(point1[0],point2[0])
min_y=min(point1[1],point2[1])
width=abs(point1[0]-point2[0])
height=abs(point1[1]-point2[1])
cut_img=img[min_y:min_y+height,min_x:min_x+width]
plt.imshow(cut_img)


