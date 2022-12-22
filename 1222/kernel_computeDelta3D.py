# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 16:31:23 2022

@author: e0947330
"""
import numpy as np

def kernel_computeDelta3D(x_delta,y_delta,z_delta,delta_f,delta_dxf,delta_dyf,delta_dzf):
	cz = 1.0
	for i in range(4):
		cy = 1.0
		for j in range(4):
			cx = 1.0
			for k in range(4):
				delta_f[i*16+j*4+k,0] = cz * cy * cx
				if(k<3):
					delta_dxf[i*16+j*4+k+1,0] = (k+1) * cz * cy * cx
				
				
				if(j<3):
					delta_dyf[i*16+(j+1)*4+k,0] = (j+1) * cz * cy * cx
				
				
				if(i<3):
					delta_dzf[(i+1)*16+j*4+k,0] = (i+1) * cz * cy * cx
				
				
				cx = cx * x_delta
			
			cy = cy * y_delta
		
		cz= cz * z_delta
        
	return delta_f,delta_dxf,delta_dyf,delta_dzf
