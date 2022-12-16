# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:54:40 2022

@author: e0947330
"""
import numpy as np


def computeDelta3Dj_v2(x_delta, y_delta, z_delta):
    

    delta_f = np.zeros((64,1))
    delta_dxf = np.zeros((64,1))
    delta_dyf = np.zeros((64,1))
    delta_dzf = np.zeros((64,1))
    delta_ddxf = np.zeros((64,1))
    delta_ddyf = np.zeros((64,1))
    delta_ddzf = np.zeros((64,1))
    cz = 1
    for i in range(4):
        
        cy = 1
        for j in range(4):
            
            cx = 1
            for k in range(4):
                
                delta_f[i*16+j*4+k] = (cx*cy*cz)
                if k<3:
                    
                    delta_dxf[i*16+j*4+k+1] = ((k+1))*cx*cy*cz
                
                if k<2:
                    
                    delta_ddxf[i*16+j*4+k+2]=((k+1)*(k+2))*cx*cy*cz
                
                if j<3:
                    
                    delta_dyf[i*16+(j+1)*4+k] = ((j+1))*cx*cy*cz
                
                if j<2:
                    
                    delta_ddyf[i*16+(j+2)*4+k]= ((j+1)*(j+2))*cx*cy*cz
                
                if i<3:
                    
                    delta_dzf[(i+1)*16+j*4+k] = ((i+1))*cx*cy*cz
                
                if i<2:
                    
                    delta_ddzf[(i+2)*16+j*4+k]=((i+1)*(i+2))*cx*cy*cz
                
                cx = cx*x_delta
            
            cy = cy*y_delta
        
        cz = cz*z_delta
    return delta_f,delta_dxf,delta_ddxf,delta_dyf,delta_ddyf,delta_dzf,delta_ddzf