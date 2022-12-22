# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 16:44:48 2022

@author: e0947330
"""
import numpy as np
from kernel_computeDelta3D import kernel_computeDelta3D
from kernel_DerivativeSpline import kernel_DerivativeSpline
from kernel_cholesky import kernel_cholesky
from kernel_luEvaluate import kernel_luEvaluate
# void kernel_splineMLEFit_z_sCMOS(const int subregion,const float *d_data,const float *d_coeff, const int spline_xsize, const int spline_ysize, const int spline_zsize, const int sz, const int iterations, 
# 	float *d_Parameters, float *d_CRLBs, float *d_LogLikelihood,float initZ, const int Nfits,const float *d_varim){
# 		/*! 
# 	 * \brief basic MLE fitting kernel.  No additional parameters are computed.
# 	 * \param d_data array of subregions to fit copied to GPU
# 	 * \param d_coeff array of spline coefficients of the PSF model
# 	 * \param spline_xsize,spline_ysize,spline_zsize, x,y,z size of spline coefficients
# 	 * \param sz nxn size of the subregion to fit
# 	 * \param iterations number of iterations for solution to converge
# 	 * \param d_Parameters array of fitting parameters to return for each subregion
# 	 * \param d_CRLBs array of Cramer-Rao lower bound estimates to return for each subregion
# 	 * \param d_LogLikelihood array of loglikelihood estimates to return for each subregion
# 	 * \param initZ intial z position used for fitting
# 	 * \param Nfits number of subregions to fit
# 	 * \param d_varim variance map of sCMOS
# 	 */


d_coeff=np.zeros((32*32*23*64,1))
for z1 in range(23):
    for j1 in range(32):
        for i1 in range(32):
            for k1 in range(64):
                d_coeff[k1+i1*64+j1*32*64+z1*32*32*64,0]=coeff[j1,i1,z1,k1]
                
Nmax=1.2
initZ=18.7
sz=33
subimg_stack=np.zeros((33,33))
subimg_stack=simpsf[:,:,0]
# data=subimg_stack.flatten()
# data=data.flatten()
s_data=subimg_stack.reshape(-1,1)
# spline_coeff=coeff
spline_xsize = coeff.shape[0]
spline_ysize = coeff.shape[1]
spline_zsize =coeff.shape[2]
size_box=33
iterations=10

# cor=np.column_stack(cor)
init_parameters=[13.5,13.8,16.7,1,0.01]#13,13,16,1,0.01

init_parameters=np.column_stack(init_parameters).T

# spline_coeff=s_coeff.flatten()
# spline_coeff=s_coeff
TOLERANCE=1e-6
ACCEPTANCE=1.5
SCALE_UP=10
SCALE_DOWN=0.1










def MLELM():	
    NV=5
    M=np.zeros((NV*NV,1))
    Diag=np.zeros((NV,1))
    Minv=np.zeros((NV*NV,1))
    

    newTheta=init_parameters
    oldTheta=init_parameters
    
    newLambda = 0.1
    oldLambda = 0.1
    mu=10
    
    
    
	
    newUpdate=[1e13,1e13,1e13,1e13,1e13]
    newUpdate=np.column_stack(newUpdate).T
    oldUpdate=[1e13,1e13,1e13,1e13,1e13]
    oldUpdate=np.column_stack(oldUpdate).T
    maxJump=[1,1,1,1,1]
    maxJump=np.column_stack(maxJump).T
    
    newdudt=np.zeros((NV,1))
    
	

    newErr = 1e12
    oldErr = 1e13

    jacobian=np.zeros((NV,1))
    hessian=np.zeros((NV*NV,1))
    
    delta_f=np.zeros((64,1))
    delta_dxf=np.zeros((64,1))
    delta_dyf=np.zeros((64,1))
    delta_dzf=np.zeros((64,1))

	
    errFlag=0;
    L=np.zeros((NV*NV,1))
    U=np.zeros((NV*NV,1))

    


    newTheta[3,0] = max(newTheta[3,0],0.01)
    newTheta[2,0]= (Nmax-newTheta[3,0])/d_coeff[int((spline_zsize/2)*(spline_xsize*spline_ysize))+int((spline_ysize/2)*spline_xsize)+int((spline_xsize/2))]*4

    newTheta[4,0]=initZ

    maxJump[2,0]=max(newTheta[2,0],maxJump[2,0])

    maxJump[3,0]=max(newTheta[3,0],maxJump[3,0])

    maxJump[4,0]= max(spline_zsize/3.0,maxJump[4,0])


    oldTheta=newTheta

	
    xc = -1.0*((newTheta[0,0]-sz)/2+0.5)
    yc = -1.0*((newTheta[1,0]-sz)/2+0.5)

    off = np.floor((spline_xsize+1.0-sz)/2)

    xstart = np.floor(xc)
    xc = xc-xstart;

    ystart = np.floor(yc)
    yc = yc-ystart;

	
    zstart = np.floor(newTheta[4,0]);
    zc = newTheta[4,0] -zstart

    newErr = 0
	
    delta_f,delta_dxf,delta_dyf,delta_dzf=kernel_computeDelta3D(xc, yc, zc, delta_f, delta_dxf, delta_dyf, delta_dzf)
    newDudt=np.zeros((5,1))
    model=0
    s_varim=np.zeros((sz*sz,1))
    for ii in range(sz):
        for jj in range(sz):
            model1,newDudt=kernel_DerivativeSpline(ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,d_coeff,newTheta,newDudt,model)
            model=model1[0]
            model +=s_varim[sz*jj+ii,0]
            data=s_data[sz*jj+ii,0]+s_varim[sz*jj+ii,0]
            
            if (data>0):
                newErr = newErr + 2*((model-data)-data*np.log(model/data))
            else:
    		
                newErr = newErr + 2*model
                data = 0
    		

            t1 = 1-data/model
            for l in range(NV):
                jacobian[l,0]+=t1*newDudt[l,0]
    		

            t2 = data/pow(model,2);
            for l in range(NV):
                  for m in range(l,NV):
                      
                      hessian[l*NV+m,0] +=t2*newDudt[l,0]*newDudt[m,0]
                      hessian[m*NV+l,0] = hessian[l*NV+m,0]
    		

    for kk in range(iterations):

            if(abs((newErr-oldErr)/newErr)<TOLERANCE):
				
                break
			
            else:
                if(newErr>ACCEPTANCE*oldErr):
					
                    for i in range(NV):
                        newTheta[i,0]=oldTheta[i,0]
                        newUpdate[i,0]=oldUpdate[i,0]
					
                    newLambda = oldLambda
                    newErr = oldErr
                    mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3)   
                    newLambda = SCALE_UP*newLambda
				
                elif(newErr<oldErr and errFlag==0):
                    newLambda = SCALE_DOWN*newLambda
                    mu = 1+newLambda
				
				

                for i in range(NV):
                    hessian[i*NV+i,0]=hessian[i*NV+i,0]*mu
				
# 				memset(L,0,NV*sizeof(float));
# 				memset(U,0,NV*sizeof(float));
                L=np.zeros((NV*NV,1))
                U=np.zeros((NV*NV,1))
                errFlag,L,U = kernel_cholesky(hessian,NV,L,U)
                if (errFlag ==0):
                    for i in range(NV):
                        oldTheta[i,0]=newTheta[i,0]
                        oldUpdate[i,0] = newUpdate[i,0]
					
                    oldLambda = newLambda
                    oldErr=newErr

                    newUpdate=kernel_luEvaluate(L,U,jacobian,NV,newUpdate)	
					
					
                    for ll in range(NV):
                        if (newUpdate[ll,0]/oldUpdate[ll,0]< -0.5):
                            maxJump[ll,0] = maxJump[ll,0]*0.5
						
                        newUpdate[ll,0] = newUpdate[ll,0]/(1+abs(newUpdate[ll,0]/maxJump[ll,0]))
                        newTheta[ll,0] = newTheta[ll,0]-newUpdate[ll,0]
					
					
                    newTheta[0,0] = max(newTheta[0,0],(sz-1)/2-sz/4.0)
                    newTheta[0,0] = min(newTheta[0,0],(sz-1)/2+sz/4.0)
                    newTheta[1,0] = max(newTheta[1,0],(sz-1)/2-sz/4.0)
                    newTheta[1,0] = min(newTheta[1,0],(sz-1)/2+sz/4.0)
                    newTheta[2,0] = max(newTheta[2,0],1.0)
                    newTheta[3,0] = max(newTheta[3,0],0.01)
                    newTheta[4,0] = max(newTheta[4,0],0.0)
                    newTheta[4,0] = min(newTheta[4,0],spline_zsize)

					
                    xc = -1.0*((newTheta[0,0]-sz/2)+0.5)
                    yc = -1.0*((newTheta[1,0]-sz/2)+0.5)

                    xstart = np.floor(xc)
                    xc = xc-xstart

                    ystart = np.floor(yc)
                    yc = yc-ystart

                    zstart = np.floor(newTheta[4,0])
                    zc = newTheta[4,0] -zstart


                    newErr = 0
# 					memset(jacobian,0,NV*sizeof(float))
# 					memset(hessian,0,NV*NV*sizeof(float))
                    jacobian=np.zeros((NV,1))
                    hessian=np.zeros((NV*NV,1))

                    delta_f,delta_dxf,delta_dyf,delta_dzf=kernel_computeDelta3D(xc, yc, zc, delta_f, delta_dxf, delta_dyf, delta_dzf)
                    newDudt=np.zeros((5,1))
                    model=0
                    for ii in range(sz):
                        for jj in range(sz):
                            model1,newDudt=kernel_DerivativeSpline(ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,d_coeff,newTheta,newDudt,model)
                            model=model1[0]
                            model +=s_varim[sz*jj+ii,0]
                            data=s_data[sz*jj+ii,0]+s_varim[sz*jj+ii,0]	

                            if (data>0):
                                newErr = newErr + 2*((model-data)-data*np.log(model/data))
                            else:
    						
                                newErr = newErr + 2*model
                                data = 0
    						
    
                            t1 = 1-data/model
                            for l in range(NV):
                                jacobian[l,0]+=t1*newDudt[l,0]
    						
    
                            t2 = data/pow(model,2)
                            for l in range(NV):
                                for m in range(l,NV):
                                    hessian[l*NV+m,0] +=t2*newDudt[l,0]*newDudt[m,0]
                                    hessian[m*NV+l,0] = hessian[l*NV+m,0]
						
					
				
                else:
				
                    mu = max( (1 + newLambda*SCALE_UP)/(1 + newLambda),1.3)        
                    newLambda = SCALE_UP*newLambda;
                    
            print(kk)
            print(newTheta) 
            print(newUpdate) 
            print(newErr) 
				

			


		
	