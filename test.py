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


input0 = nd2.imread('G:/tony lab/cx/E 488 off 561 on 001.nd2') # read to numpy array
size = input0.shape
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
input6=np.ones((size[0],size[1],size[2])).astype(np.uint8)

for i in range(size[0]):
    input1max=np.max(np.hstack(input0[i,:,:]))
    input5=255*input0[i,:,:]/input1max
    input6[i,:,:]=input5.astype(np.uint8)


def mouse1(event,x,y,flags,param):
    global zoom,runimage,flag1,RECORDZOOM
    flag1=0
    text="NUM:"+str(zoom)
    runimage=input6[zoom,:,:]     
    cv2.putText(runimage,text,(100,100),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2)
    cv2.imshow('image',runimage)
    if event == cv2.EVENT_MOUSEWHEEL:  # 滚轮
       
       if  zoom==0:
             if flags > 0:  # 滚轮上移
                 zoom += wheel_step
                 # runimage=input6[zoom,:,:]
             else:  # 滚轮下移
                 zoom -= 0   
       elif  zoom==(size[0]-1):
              if flags < 0:  # 
                  zoom -= wheel_step
                  # runimage=input6[zoom,:,:]
              else:  # 
                  zoom -= 0   
       else:
                       if flags > 0:  # 滚轮上移
                           zoom += wheel_step
                           # runimage=input6[zoom,:,:]
                       else:  # 滚轮下移
                           zoom -= wheel_step
    elif event == cv2.EVENT_MBUTTONDOWN:
        RECORDZOOM[flag1]=zoom
        flag1 += 1
        
    else:
        zoom -= 0
   
    
    # else:
    #      cv2.putText(runimage,text,(100,500),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2)
    #      cv2.imshow('image',runimage)
     
wheel_step, zoom = 1, 0  # 滚轮值，与初始值
RECORDZOOM=[0,1]
runimage=input6[0,:,:]
cv2.namedWindow('image')
cv2.setMouseCallback('image',mouse1)
cv2.waitKey(0)



def mouse2(event,x,y,flags,param):
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
      


wheel_step, zoom = 1, 0  # 缩放系数， 缩放值

cv2.namedWindow('image')
cv2.setMouseCallback('image',mouse2)
cv2.imshow('image',img)
cv2.waitKey(0)

min_x=min(point1[0],point2[0])
min_y=min(point1[1],point2[1])
width=abs(point1[0]-point2[0])
height=abs(point1[1]-point2[1])
cut_img=img[min_y:min_y+height,min_x:min_x+width]
plt.imshow(cut_img)


