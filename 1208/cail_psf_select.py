# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:08:28 2022

@author: e0947330
"""

import tifffile
import numpy as  np
from scipy.interpolate import griddata,interp1d
import matplotlib.pyplot as plt
import cv2
import math
import nd2
from scipy import signal


from scipy import signal


input0= nd2.imread('G:/tony lab/cx/PSF_CAIL/1108 beads008.nd2') # read to numpy array

x_pixel=0.065
y_pixel=0.065#um
z_step_size=0.1
total_photon=1




# x_pixel=0.065
# y_pixel=0.065#um
# z_step_size=0.1
# total_photon=4000

input0_size = input0.shape
input0_x_size=input0_size[2]
input0_y_size=input0_size[1]
input0_z_size=input0_size[0]


input0_255=np.zeros((input0_z_size,input0_y_size,input0_x_size)).astype(np.uint8)
for i0 in range(input0_z_size):
    
    input0_255[i0,:,:]=(255*input0[i0,:,:]/np.max(input0[i0,:,:])).astype(np.uint8)
input0_255=input0_255.astype(np.uint8)


# cv2.namedWindow("image")
# cv2.imshow("image", input0_255[0,:,:])
# cv2.waitKey(0)
def Frame_selection(windowname, img0):
    
    def mouse1(event,x,y,flags,param): #Use the mouse wheel to browse the picture stack, and use the middle click to achieve the extraction of the image sequence number of interest
        
        global zoom,runimage,flag1,RECORDZOOM
        
        text="NUM:"+str(zoom)
        runimage=img0[zoom,:,:]     
        cv2.putText(runimage,text,(100,100),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2)
        cv2.imshow(windowname,runimage)
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
   

     
    
    size=img0.shape
    cv2.namedWindow(windowname)
    cv2.setMouseCallback(windowname,mouse1) #Browse the picture and select the picture
    cv2.waitKey(0)

    lengthimg=RECORDZOOM[1]-RECORDZOOM[0] #The size of the selected image sequence
    return [RECORDZOOM[0],RECORDZOOM[1]]


wheel_step, zoom = 1, 0  # Mouse wheel step value and initial value
flag1=0
RECORDZOOM=[0,1] #Stores the selected sequence numbers
runimage=input0_255[0,:,:]
[start_frame,final_fram]=Frame_selection('image0', input0_255)





def SetPoints(windowname, img):
    """
    输入图片，打开该图片进行标记点，返回的是标记的几个点的字符串
    """
    print('(提示：单击需要标记的坐标，Enter确定，Esc跳过，其它重试。)')
    points = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(temp_img, (x, y), 2, (255, 0, 0), -1)
            points.append([x, y])
            cv2.imshow(windowname, temp_img)

    temp_img = img.copy()
    cv2.namedWindow(windowname)
    cv2.setMouseCallback(windowname, onMouse)
    cv2.imshow(windowname, temp_img)
    cv2.waitKey(0)
    key = cv2.waitKey(0)
    if key == 13:  # Enter
        print('坐标为：', str(points))
        del temp_img
        cv2.destroyAllWindows()
        return points
    elif key == 27:  # ESC
        print('跳过该张图片')
        del temp_img
        cv2.destroyAllWindows()
        return
    else:
        print('重试!')
        return SetPoints(windowname, img)


zuobiao=SetPoints("image1",input0_255[start_frame,:,:])

win_size_small=33
half_win_size_small=int(win_size_small/2)
cail_psf=np.zeros((final_fram-start_frame+1,win_size_small,win_size_small))
for i1 in range(int(np.size(zuobiao)/2)):
    zuobiao0=zuobiao[i1]
    single_psf_denoise_total=np.zeros((final_fram-start_frame+1,win_size_small,win_size_small))
    for i2 in range(start_frame,final_fram+1):
        single_psf=input0[i2,zuobiao0[1]-half_win_size_small:zuobiao0[1]+half_win_size_small+1,zuobiao0[0]-half_win_size_small:zuobiao0[0]+half_win_size_small+1]
        
        single_psf_denoise=single_psf-np.sum(single_psf[0:10,0:10]+single_psf[0:10,win_size_small-11:win_size_small-1]+single_psf[win_size_small-11:win_size_small-1,0:10]++single_psf[win_size_small-11:win_size_small-1,win_size_small-11:win_size_small-1])/(win_size_small*win_size_small)
        single_psf_denoise=np.where(single_psf_denoise > 0, single_psf_denoise, 0)
        single_psf_denoise_total[i2-start_frame,:,:]=single_psf_denoise
    cail_psf+=single_psf_denoise_total   

cail_psf=cail_psf/(np.size(zuobiao)/2)

for i1 in range(cail_psf.shape[0]):
    
    cail_psf[i1,:,:]=cail_psf[i1,:,:]/np.sum(cail_psf[i1,:,:])
    sum_cail_psf=np.sum(cail_psf[i1,:,:])
    cail_psf[i1,:,:]= cail_psf[i1,:,:]-1*sum_cail_psf/(win_size_small*win_size_small)
    cail_psf[i1,:,:]=np.where(cail_psf[i1,:,:] >0, cail_psf[i1,:,:], 0)