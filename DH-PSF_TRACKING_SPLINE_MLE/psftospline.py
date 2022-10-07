# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:49:18 2022

@author: e0947330
"""
import tifffile
import numpy as  np
from scipy.interpolate import griddata,interp1d
import matplotlib.pyplot as plt




input0 = np.array(tifffile.imread('G:/tony lab/cx/test.tif'),dtype=float)
# x_pixel=
# y_pixel=
z_step_size=0.1
input0_size = input0.shape
input0_x_size=input0_size[2]
input0_y_size=input0_size[1]
input0_z_size=input0_size[0]
upsample=4
grid_x, grid_y = np.mgrid[0:1:((upsample*input0_x_size)-(upsample-1))*1j, 0:1:((upsample*input0_y_size)-(upsample-1))*1j]

points =  np.mgrid[0:1:input0_x_size*1j, 0:1:input0_y_size*1j]
x_points=points[0,:].reshape(-1, 1)
y_points=points[1,:].reshape(-1, 1)
xy_points=np.append(x_points,y_points,axis=1)
spline_xy=np.zeros(((upsample*input0_x_size)-(upsample-1),(upsample*input0_y_size)-(upsample-1),input0_z_size))
for i in range(input0_z_size):
    
    values_points=input0[i,:].reshape(-1, 1)
    grid_z2 = griddata(xy_points, values_points, (grid_x, grid_y), method='cubic')
    # plt.imshow(grid_z2[:,:,0], origin='lower')
    spline_xy[:,:,i]=grid_z2[:,:,0]
    
spline_xy_size = spline_xy.shape
spline_xy_y_size=spline_xy_size[0]
spline_xy_x_size=spline_xy_size[1]
 
z_points=np.linspace(0,z_step_size*(input0_z_size-1),input0_z_size)
grid_z=np.linspace(0,z_step_size*(input0_z_size-1),((upsample*input0_z_size)-(upsample-1)))
spline_xyz=np.zeros(((upsample*input0_x_size)-(upsample-1),(upsample*input0_y_size)-(upsample-1),(upsample*input0_z_size)-(upsample-1)))
for i1 in range(spline_xy_y_size):
    for j1 in range(spline_xy_x_size):
        z_values=spline_xy[i1,j1,:]
        f2 = interp1d(z_points, z_values, kind='cubic')
        spline_xyz[i1,j1,:]=f2(grid_z)
        plt.imshow(spline_xyz[:,:,36], origin='lower')
        plt.imshow(input0[9,:,:], origin='lower')
        