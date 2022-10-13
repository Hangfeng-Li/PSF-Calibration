# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 16:54:38 2022

@author: e0947330
"""
import tifffile
import cv2
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import griddata,interp1d

input0 = np.array(tifffile.imread('G:/tony lab/cx/E 488 off 561 on 001-half.tif'))
input0_size = input0.shape
img=input0[10,:,:] 
imgmax=np.max(np.hstack(img))
img1=255*img/imgmax
img2=img1.astype(np.uint8)
cv2.namedWindow('image2')
cv2.imshow('image2',img2)
cv2.waitKey(0)
#
def mouse3(event,x,y,flags,param): #Capture the area of interest in the picture
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
cv2.setMouseCallback('image2',mouse3) #Capture images 
cv2.imshow('image2',img2)
cv2.waitKey(0)

min_x=min(point1[0],point2[0]) #The two points selected in the picture (top left point and bottom right point)
min_y=min(point1[1],point2[1])
width=abs(point1[0]-point2[0])
height=abs(point1[1]-point2[1])

cut_img=img[min_y:min_y+height,min_x:min_x+width] #A cropped image collection

cut_img_xsize=cut_img.shape[1]
cut_img_ysize=cut_img.shape[0]


psf_nobacknoise=np.zeros((cut_img.shape))

backnoise=np.sum(cut_img)/(cut_img_xsize*cut_img_ysize)
psf_nobacknoise=np.rint(img-backnoise*np.ones((np.shape(img))))
    
psf_nobacknoise[psf_nobacknoise < 0] = 0

#


diameter_psf=14#pixel
distance_psf=18#pixel
big_circel=diameter_psf+distance_psf
small_circel=distance_psf-diameter_psf
mask0=np.zeros((big_circel,big_circel))


if big_circel%2==0:
    centre_circel=(big_circel-1)/2.0
    for i1 in range(big_circel):
        for j1 in range(big_circel):
            if np.square(i1-centre_circel)+np.square(j1-centre_circel)<=np.square(big_circel/2.0):
                mask0[i1,j1]=1
else:
    centre_circel=big_circel/2.0
    for i1 in range(big_circel):
        for j1 in range(big_circel):
            if np.square(i1-centre_circel)+np.square(j1-centre_circel)<=np.square(big_circel/2.0):
                mask0[i1,j1]=1
                
plt.imshow(mask0, origin='lower')                  
                
if small_circel%2==0:
    centre_circel=(big_circel-1)/2.0
    for i1 in range(big_circel):
        for j1 in range(big_circel):
            if np.square(i1-centre_circel)+np.square(j1-centre_circel)<=np.square(small_circel/2.0):
                mask0[i1,j1]=0
else:
    centre_circel=big_circel/2.0
    for i1 in range(big_circel):
        for j1 in range(big_circel):
            if np.square(i1-centre_circel)+np.square(j1-centre_circel)<=np.square(small_circel/2.0):
                mask0[i1,j1]=0
    
plt.imshow(mask0, origin='lower')            

mask0=mask_extract2

scharr=mask0




grad=signal.convolve2d(psf_nobacknoise,scharr,boundary='symm',mode='same') 
plt.imshow(input0[10,:,:], origin='lower')
plt.imshow(grad, origin='lower')
grad0=((grad>20) )
grad00=grad0*np.ones(grad0.shape)
plt.imshow(grad0, origin='lower')

# kernel_w=int((distance_psf+diameter_psf)/2)
# kernel = np.ones((kernel_w,kernel_w),np.uint8)
# grad01 = cv2.dilate(grad00,kernel,iterations = 4)
# grad01 = cv2.erode(grad01,kernel,iterations = 3)
# grad1=grad01.astype(np.uint8)*255
# cv2.namedWindow('GRAD1')
# cv2.imshow('GRAD1',grad1)
# cv2.waitKey(0)

# text11=(np.multiply(psf_nobacknoise,grad01))
# text12=np.array(text11,dtype=int)
# plt.imshow(text12, origin='lower')

# img12=text12
# imgmax=np.max(np.hstack(img12))
# img13=255*img12/imgmax
# img14=img13.astype(np.uint8)

# cv2.namedWindow('image4')
# cv2.imshow('image4',img14)
# cv2.waitKey(0)
# cv2.namedWindow('grad01')
# cv2.imshow('grad01',grad01)
# cv2.waitKey(0)

grad11=grad00.astype(np.uint8)
num_labels, labels, stats, centroids=cv2.connectedComponentsWithStats(grad11,connectivity=8)
for i1 in range(stats.shape[0]):
    if stats[i1,4]<50 or stats[i1,4]>300:
       labels[labels==i1]=0 
labels_mask=np.zeros(labels.shape)

labels_mask[labels>0]=1
plt.imshow(labels_mask, origin='lower')

text111=(np.multiply(psf_nobacknoise,labels_mask))
text121=np.array(text111,dtype=int)
img121=text121
imgmax=np.max(np.hstack(img121))
img131=255*img121/imgmax
img141=img131.astype(np.uint8)
cv2.namedWindow('image141')
cv2.imshow('image141',img141)
cv2.waitKey(0)

num_labels_mask, labels_mask, stats_mask, centroids_mask=cv2.connectedComponentsWithStats(labels_mask.astype(np.uint8),connectivity=8)
new_mask=np.zeros(labels_mask.shape)
for i1 in range(num_labels_mask):
    if (math.ceil(big_circel/2)<=centroids_mask[i1,0]) and (math.ceil(big_circel/2)<=centroids_mask[i1,1]) and (input0_size[2]-centroids_mask[i1,0]>math.ceil(big_circel/2)) and (input0_size[1]-centroids_mask[i1,1]>math.ceil(big_circel/2)):
       y_centroids_mask=math.ceil(centroids_mask[i1,0]) 
       x_centroids_mask=math.ceil(centroids_mask[i1,1]) 
       # new_mask[(y_centroids_mask-math.ceil(big_circel/2)):(y_centroids_mask+math.ceil(big_circel/2)),(x_centroids_mask-math.ceil(big_circel/2)):(x_centroids_mask+math.ceil(big_circel/2))]=1
       new_mask[(x_centroids_mask-math.ceil(big_circel/2)):(x_centroids_mask+math.ceil(big_circel/2)),(y_centroids_mask-math.ceil(big_circel/2)):(y_centroids_mask+math.ceil(big_circel/2))]=1
plt.imshow(new_mask, origin='lower')
text111=(np.multiply(psf_nobacknoise,new_mask))
text121=np.array(text111,dtype=int)
img121=text121
imgmax=np.max(np.hstack(img121))
img131=255*img121/imgmax
img141=img131.astype(np.uint8)
cv2.namedWindow('image141')
cv2.imshow('image141',img141)
cv2.waitKey(0)




num_new_mask, labels_new_mask, stats_new_mask, centroids_new_mask=cv2.connectedComponentsWithStats(new_mask.astype(np.uint8),connectivity=8)
input0_x_size=int(math.sqrt(stats_new_mask[1,4]))
input0_y_size=input0_x_size
upsample=4
calib_size=125########
padzero=calib_size-((upsample*input0_x_size)-(upsample-1))
total_photon=4000
record_z_min=np.zeros((num_new_mask,2))
spline_xy_padzeros_photon=np.zeros(((upsample*input0_x_size)-(upsample-1),(upsample*input0_y_size)-(upsample-1),num_new_mask))
corr=np.zeros(spline_xy_padzeros_photon.shape)
ycorr=np.zeros(num_new_mask)
xcorr=ycorr
for i1 in range(num_new_mask):
    # mask_sep=np.zeros((input0_x_size,input0_y_size))
    if i1>0:
        # mask_sep=np.where(labels_new_mask==i1)
        grid_x, grid_y = np.mgrid[0:input0_x_size:((upsample*input0_x_size)-(upsample-1))*1j, 0:input0_y_size:((upsample*input0_y_size)-(upsample-1))*1j]
    
        points =  np.mgrid[0:input0_x_size:input0_x_size*1j, 0:input0_y_size:input0_y_size*1j]
        x_points=points[0,:].reshape(-1, 1)
        y_points=points[1,:].reshape(-1, 1)
        xy_points=np.append(x_points,y_points,axis=1)
        spline_xy=np.zeros(((upsample*input0_x_size)-(upsample-1),(upsample*input0_y_size)-(upsample-1)))   
        values_points0=img[stats_new_mask[i1,1]:stats_new_mask[i1,1]+input0_y_size,stats_new_mask[i1,0]:stats_new_mask[i1,0]+input0_x_size]
        mask_image=np.zeros(values_points0.shape)
        total_energy=np.sum(values_points0)
        avage_energy=total_energy/(input0_x_size*input0_y_size)
        input0_nonoise=values_points0-avage_energy*0.2
        input0_nonoise[:,:][input0_nonoise < 0] = 0
        values_points=input0_nonoise.reshape(-1, 1)
        grid_z2 = griddata(xy_points, values_points, (grid_x, grid_y), method='cubic')
        spline_xy=grid_z2[:,:,0]
        spline_xy_padzeros=np.pad(spline_xy,((int(padzero/2),int(padzero/2)),(int(padzero/2),int(padzero/2))),'constant', constant_values=(0,0)) 
        sumz=np.zeros(fft2_spline_xyz.shape[2])
        spline_xy_padzeros_photon[:,:,i1]=total_photon*spline_xy_padzeros/np.sum(spline_xy_padzeros)
        for j1 in range(fft2_spline_xyz.shape[2]):
            
            # fft2_spline_xy_padzeros_photon=np.fft.fftshift(np.fft.fft2(spline_xy_padzeros_photon))
            fft2_spline_xy_padzeros_photon=np.fft.fft2(spline_xy_padzeros_photon[:,:,i1])
            
            aa=4000*np.absolute(fft2_spline_xy_padzeros_photon)/np.sum(np.absolute(fft2_spline_xy_padzeros_photon))
            bb=4000*np.absolute(fft2_spline_xyz[:,:,j1])/np.sum(np.absolute(fft2_spline_xyz[:,:,j1]))
            # sumz[j1]=np.sum((np.absolute(fft2_spline_xy_padzeros_photon)-np.absolute(fft2_spline_xyz[:,:,j1]))**2)
            sumz[j1]=np.sum((aa-bb)**2)
        sumz_list=sumz.tolist()
        sumz_min_list = min(sumz_list) 
        min_index = sumz_list.index(min(sumz_list)) 
        
        record_z_min[i1,:]=[sumz_min_list,min_index]
        corr[:,:,i1] = signal.correlate2d( spline_xyz_photon[:,:,min_index],spline_xy_padzeros_photon[:,:,i1], boundary='symm', mode='same')
        ycorr[i1], xcorr[i1] = np.unravel_index(np.argmax(corr[:,:,i1]), corr[:,:,i1].shape)
    
# plt.imshow(np.absolute(fft2_spline_xyz[:,:,80]), origin='lower')
plt.imshow(spline_xyz[:,:,12],origin='lower')
plt.scatter(xcorr[6], ycorr[6],s=10,c="r")
# plt.imshow(np.absolute(fft2_spline_xy_padzeros_photon), origin='lower')
plt.imshow(spline_xy_padzeros_photon[:,:,6], origin='lower')
# plt.imshow(fft2_spline_xy_padzeros_photon, origin='lower')
plt.imshow(np.absolute(corr), origin='lower')


from skimage import io
io.imsave('G:/tony lab/cx/change.tif', text12)

