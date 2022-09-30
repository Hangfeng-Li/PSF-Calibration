# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:23:53 2022

@author: Li Hangfeng
"""

import nd2
import numpy as np
from PIL import Image
from matplotlib.widgets import Cursor
import cv2
import matplotlib.pyplot as plt
import gausscenter
import math
import heapq
from tracker import *

import tifffile
global img
global point1,point2


# input0 = nd2.imread('G:/tony lab/cx/E 488 off 561 on 001.nd2') # read to numpy array
input0 = tifffile.imread('G:/tony lab/cx/E 488 off 561 on 001-half.tif')
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
        cv2.circle(img2,point1,10,(0,255,0),2)
        cv2.imshow('image2',img2)

    elif event==cv2.EVENT_MOUSEMOVE and (flags&cv2.EVENT_FLAG_LBUTTON): #Move the mouse and drag with the left button
        cv2.rectangle(img2,point1,(x,y),(255,0,0),2)
        cv2.imshow('image2',img2)

    elif event==cv2.EVENT_LBUTTONUP: #Left-click to release
        point2=(x,y)
        cv2.rectangle(img2,point1,point2,(0,0,255),2)
        cv2.imshow('image2',img2)
      




cv2.namedWindow('image2')
cv2.setMouseCallback('image2',mouse2) #Capture images 
# cv2.imshow('image2',img)
cv2.waitKey(0)

min_x=min(point1[0],point2[0]) #The two points selected in the picture (top left point and bottom right point)
min_y=min(point1[1],point2[1])
width=abs(point1[0]-point2[0])
height=abs(point1[1]-point2[1])

cut_img=input0[RECORDZOOM[0]:RECORDZOOM[1],min_y:min_y+height,min_x:min_x+width] #A cropped image collection


def mouse3(event,x,y,flags,param): #Select the background noise area
    global copyorignimg,point3,point4
    
    img4=copyorignimg.copy()
    imgmax=np.max(np.hstack(img4))
    img00=255*img4/imgmax
    orign_image_uint8=img00.astype(np.uint8)
    gray_img00 = orign_image_uint8
    cv2.imshow('gray_img00',gray_img00)
    if event==cv2.EVENT_LBUTTONDOWN: #Left click
        point3=(x,y)
        cv2.circle(img4,point3,10,(0,255,0),2)
        cv2.imshow('gray_img00',gray_img00)

    elif event==cv2.EVENT_MOUSEMOVE and (flags&cv2.EVENT_FLAG_LBUTTON): #Move the mouse and drag with the left button
        cv2.rectangle(img4,point3,(x,y),(255,0,0),2)
        cv2.imshow('gray_img00',gray_img00)

    elif event==cv2.EVENT_LBUTTONUP: #Left-click to release
        point4=(x,y)
        cv2.rectangle(img4,point3,point4,(0,0,255),2)
        cv2.imshow('gray_img00',gray_img00)




tracker = EuclideanDistTracker() #Tracker of choice,The distance is used to judge whether it is the same target
record={}

# tracker_type = 'MIL'  
# tracker = cv2.MultiTracker_create()
for i in range(lengthimg):
    
# orign_image=cv2.imread('G:/tony lab/cx/Capture.png')
    orign_image=cut_img[i,:,:]
    if i==0: #Select the background noise area
        
        
        
        
        copyorignimg=orign_image
        cv2.namedWindow('gray_img00')
        cv2.setMouseCallback('gray_img00',mouse3) #Use the mouse3 function 
        cv2.imshow('gray_img00',orign_image)
        cv2.waitKey(0)

        min_x3=min(point3[0],point4[0]) #The two points selected in the image (top left point and bottom right point)
        min_y3=min(point3[1],point4[1])
        width3=abs(point3[0]-point4[0])
        height3=abs(point3[1]-point4[1])

    cut_imgnoise=orign_image[min_y3:min_y3+height3,min_x3:min_x3+width3] #the background noise area

    cut_imgnoise_xsize=cut_imgnoise.shape[0]
    cut_imgnoise_ysize=cut_imgnoise.shape[1]

        
        
    backnoise=np.sum(cut_imgnoise)/(cut_imgnoise_xsize*cut_imgnoise_ysize)
    psf_nobacknoise=np.rint(orign_image-backnoise*np.ones((np.shape(orign_image))))
    
    psf_nobacknoise[psf_nobacknoise < 0] = 0 #Image after removing the average background noise
    orign_image=psf_nobacknoise
    
    
    
    coor=gausscenter.gc(orign_image) #Use Gaussian Fitting
    y=coor[:,0]
    x=coor[:,1] #The coordinates of the spot
    
    detections = []
    for count in range(len(y)):
        detections.append([x[count], y[count]])
        
    boxes_ids = tracker.update(detections) #Label the spot of light
    # recordframe=[]
    # for box_id in boxes_ids:
    #     x1, y1, id = box_id
    #     recordframe[id]=(x1,y1)
        
    record[i]=boxes_ids
    
    if i==0:
        pair={} #Pair the spots of light
        record0=record[0]
        a = np.linspace(0,len(boxes_ids)-1,len(boxes_ids))
        a1=a.astype(np.int)
        b = []  
        
        for j in range(1, len(boxes_ids)):  
            b += zip(a1[:-j], a1[j:]) #The arrangement and combination of spots.such as,(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
            
        distance=np.ones(len(b)) #The distance between two spots in each permutation
        for j1 in range(len(b)):
            index=np.array(b[j1])
            x2,y2,id2=record0[index[0]] 
            x3,y3,id3=record0[index[1]]
            distance[j1] = math.hypot(x2 - x3, y2 - y3)
            
        
        pairnum=int(len(boxes_ids)/2) 
        idx = np.argpartition(distance, pairnum)[0:pairnum] #Sort the distances and take the  minimums(num=pair)
        pair={}
        for j2 in range(len(idx)): #The center coordinates of DH-PSF are extracted and stored
            pairseparate=np.array(b[idx[j2]])
            xc1,yc1,idc1=boxes_ids[pairseparate[0]]
            xc2,yc2,idc2=boxes_ids[pairseparate[1]]
            xc=(xc1+xc2)/2
            yc=(yc1+yc2)/2
            pair[j2]=(xc,yc,b[idx[j2]])
    
    else:#Non-first frame image processing process
        
        record0=record[i]
        a = np.linspace(0,len(boxes_ids)-1,len(boxes_ids))
        a1=a.astype(np.int)
        ridx={}  
        
        
        for j in range(len(pair)): 
            
            prepair=np.array(pair[j])
            paircompair=np.zeros(len(boxes_ids))
            for j1 in range(len(boxes_ids)):#The distance between each spot and the DH-PSF center of the previous frame
                
                x2=np.array(boxes_ids[j1])[0]
                y2=np.array(boxes_ids[j1])[1]
                distancec = math.hypot(x2 - prepair[0], y2 - prepair[1])
                paircompair[j1]=distancec
                
            # num=int(len(pair))
            num=2 #Select the two points closest to the DH-PSF center in the previous frame
            idx = np.argpartition(paircompair, num)[0:num]#
            ridx[j]=idx
        
        for j2 in range(len(pair)): #The center of DH-PSF of this frame is calculated
            xc1,yc1,idc1=boxes_ids[ridx[j2][0]]
            xc2,yc2,idc2=boxes_ids[ridx[j2][1]]
            xc=(xc1+xc2)/2
            yc=(yc1+yc2)/2
            pair[j2]=(xc,yc,(ridx[j2][0],ridx[j2][1]))
            
    imgmax=np.max(np.hstack(orign_image))
    img1=255*orign_image/imgmax
    orign_image_uint8=img1.astype(np.uint8)
    gray_img = orign_image_uint8   
    xtl=np.zeros(len(pair))  
    ytl=np.zeros(len(pair)) 
    idi=np.zeros(len(pair)) 
    
    angle=np.zeros(len(pair)) 
    
    for i2 in range(len(pair)):
        x12c=np.zeros(2)
        y12c=np.zeros(2)
        
        for i3 in range(2):
            x12c[i3]=boxes_ids[pair[i2][2][i3]][0]
            y12c[i3]=boxes_ids[pair[i2][2][i3]][1]
            
        angle[i2]=math.atan((x12c[0]-x12c[1])/(y12c[0]-y12c[1])) #Computing Angle
        if angle[i2]<0:
             angle[i2]=angle[i2]+(math.pi) #Computing Angle
        # if angle[i2]>3:
        #     angle[i2]=2*angle[i2]-(math.pi)
            
        xc= pair[i2][0]
        yc= pair[i2][1]
        abx= abs(x12c[0]-x12c[1])
        aby= abs(y12c[0]-y12c[1]) 
        xtl[i2]=int(xc-(abx+20)/2) #The upper-left corner coordinates of the box
        ytl[i2]=int(yc-(aby+20)/2)
        w=int(abx+20) #Choose the wide x length
        h=int(aby+20) #Select the wide x width
        idi[i2]=int(i2+1)
        text = "ID:"+str(idi[i2])+" " + "Angle:"+str(round(angle[i2],2))   
       
        cv2.putText(gray_img, text, (int(xtl[i2]-20), int(ytl[i2] - 5)), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0), 1)
        cv2.rectangle(gray_img, (int(xtl[i2]), int(ytl[i2])), (int(xtl[i2]+w), int(ytl[i2]+h)), (255, 0, 0), 2) 
        
    cv2.imshow("gray_img", gray_img)
    cv2.waitKey(0)
    plt.figure()
    plt.imshow(orign_image)
    plt.scatter(x,y,s=10,c="r")
    y=coor[:,0]
    x=coor[:,1]   
    cv2.imwrite('G:/tony lab/cx/ppt/%d.png' %(i), gray_img) #save images  
    




    













