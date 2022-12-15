# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 23:09:09 2022

@author: lihsn
"""
import numpy as np



def computeDelta3D(x_delta,y_delta,z_delta,delta_f,delta_dxf,delta_dyf,delta_dzf):

    cz=1
    for i in range(4):
        cy=1
        for j in range(4):
            cx=1
            for k in range(4):
                delta_f[i*16+j*4+k+1]=cz*cy*cx
                if k<3:
                    delta_dxf[i*16+j*4+k+1]=(k+1)*cz*cy*cx
                if j<3:
                    delta_dyf[i*16+j*4+k+1]=(j+1)*cz*cy*cx
                if i<3:
                    delta_dzf[i*16+j*4+k+1]=(i+1)*cz*cy*cx
                cx=cx*x_delta
            cy=cy*y_delta
        cz=cz*z_delta
    return delta_f,delta_dxf,delta_dyf,delta_dzf