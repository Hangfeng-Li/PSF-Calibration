# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:15:09 2022

@author: e0947330
"""
import numpy as np
import cv2
import math

def gc(orign_image):
    
    
    # orign_image=cv2.imread('G:/tony lab/cx/Capture.png')
    imgmax=np.max(np.hstack(orign_image))
    img1=255*orign_image/imgmax
    orign_image_uint8=img1.astype(np.uint8)
    gray_img = orign_image_uint8
    # gray_img = cv2.cvtColor(orign_image_uint8, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('gray_img')
    cv2.imshow('gray_img',gray_img)
    cv2.waitKey(0)
    
    blur_image=cv2.GaussianBlur(gray_img,[5,5],0)
    cv2.namedWindow('blur_image')
    cv2.imshow('blur_image',blur_image)
    cv2.waitKey(0)
    
    ret,mask_image=cv2.threshold(blur_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)  
    erosion = cv2.erode(mask_image,kernel,iterations = 1)
    kernel = np.ones((5,5),np.uint8) 
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    mask_image=dilation
    ret,labels,stats,centroid=cv2.connectedComponentsWithStats(mask_image)
    cv2.namedWindow('mask_image')
    cv2.imshow('mask_image',mask_image)
    cv2.waitKey(0)
    
    coor=np.zeros((ret-1,2))
    for i in range(ret-1):
       x,y=np.where(labels==(i+1))
       length=len(x)
       temp_A=np.zeros((length,1))
       temp_B=np.zeros((length,5))
       for j in range(length):
           psrc=orign_image[x[j],y[j]].astype("float32")
           if psrc>0:
              temp_A[j]=psrc*math.log(psrc)
               
           temp_B[j,0]=psrc
           temp_B[j,1]=psrc*x[j]
           temp_B[j,2]=psrc*y[j]
           temp_B[j,3]=psrc*x[j]*x[j]
           temp_B[j,4]=psrc*y[j]*y[j]
    
       Vector_A=temp_A
       matrix_B=temp_B
    
       # def SchmitOrth(matrix_B:np.array):
       #      cols = matrix_B.shape[1]

       #      Q = np.copy(matrix_B)
       #      R = np.zeros((cols, cols))

       #      for col in range(cols):
       #          for k in range(col):
       #              k =  np.sum(matrix_B[:, col] * Q[:, k]) / np.sum( np.square(Q[:, k]) )
       #              Q[:, col] -= k*Q[:, k]
       #              Q[:, col] /= np.linalg.norm(Q[:, col])

       #          for l in range(cols):
       #              R[col, l] = Q[:, col].dot( matrix_B[:, l])

       #      return Q, R

       
       [Q,R]=np.linalg.qr(matrix_B)
       S=np.dot(Q.T,Vector_A)
       S=S[0:5]
       R1=R[0:5,0:5]
       C=np.linalg.solve(R1,S)
       coor[i,:]=[-0.5*C[1]/C[3],-0.5*C[2]/C[4]]
    return coor