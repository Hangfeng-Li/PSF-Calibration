# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 23:11:14 2022

@author: lihsn
"""

import numpy as np
from computeDelta3D import computeDelta3D
from DerivativeSpline import DerivativeSpline
from choolesky import choolesky
from luEvaluate import luEvaluate
from DerivativeSpline_v2 import DerivativeSpline_v2


s_coeff=coeff
subimg_stack=np.zeros((33,33))
subimg_stack=simpsf[:,:,0]
# data=subimg_stack.flatten()
# data=data.flatten()
data=subimg_stack.flatten()
# spline_coeff=coeff
spline_xsize = coeff.shape[0]
spline_ysize = coeff.shape[1]
spline_zsize =coeff.shape[2]
size_box=33
iterations=30

# cor=np.column_stack(cor)
init_parameters=[13.5,13.8,16.4,0.01,0.009]
init_parameters=np.column_stack(init_parameters).T

spline_coeff=s_coeff.flatten()





def SplineMLE(subimg_stack, data,spline_coeff,spline_xsize,spline_ysize,spline_zsize,size_box,iterations,init_parameters):
    
    NV=5
    M=np.zeros((NV*NV,1))
    Diag=np.zeros((NV,1))
    Minv=np.zeros((NV*NV))
    
    xstart=init_parameters[0,0]
    ystart=init_parameters[1,0]
    zstart=init_parameters[2,0]
    Nstart=init_parameters[3,0]
    Bgstart=init_parameters[4,0]
    
    
    xc=xstart
    yc=ystart
    zc=zstart
    
    
    
    newlambda=0.1
    oldlambda=0.1
    newupdate=[1e13,1e13,1e13,1e13,1e13]
    newupdate=np.column_stack(newupdate).T
    oldupdate=[1e13,1e13,1e13,1e13,1e13]
    oldupdate=np.column_stack(oldupdate).T
    maxjump=[1,1,2,100,20]
    maxjump=np.column_stack(maxjump).T
    newerr=1e12
    olderr=1e13
    tolerance=1e-6
    scaleup=10
    scaledown=0.1
    acceptance=1.5
    
    
    
    newdudt=np.zeros((NV,1))
    jacobian=np.zeros((NV,1))
    hessian=np.zeros((NV*NV,1))
    delta_f=np.zeros((64,1))
    delta_dxf=np.zeros((64,1))
    delta_dyf=np.zeros((64,1))
    delta_dzf=np.zeros((64,1))
    errflag=0
    L=np.zeros((NV*NV,1))
    U=np.zeros((NV*NV,1))
    
    Nmax=np.max(subimg_stack)
    newtheta=init_parameters
    newtheta[4]=max([newtheta[4],0.01])
    # newtheta[3]=(Nmax-newtheta[4])/spline_coeff[int((spline_zsize/2)*(spline_xsize*spline_ysize))+int((spline_ysize/2)*spline_xsize)+int((spline_xsize/2))]*4
    #(x,y,bg,I,z)
    maxjump[3]=max([newtheta[3],maxjump[3]])
    maxjump[4]=max([newtheta[4],maxjump[4]])
    maxjump[2]=max([spline_zsize/3,maxjump[2]])
    
    oldtheta=newtheta
    
    xc=-1*(newtheta[0]-size_box/2+0.5)
    yc=-1*(newtheta[1]-size_box/2+0.5)
    
    off=np.floor((spline_xsize+1-size_box)/2)
    
    xstart=np.floor(xc)
    xc=xc-xstart
    
    ystart=np.floor(yc)
    yc=yc-ystart
    
    zstart=np.floor(newtheta[2])
    zc=newtheta[2]-zstart
    
    newerr=0
    
    ####??????xc???yc???zc?????????delta_f,delta_dxf,delta_dyf,delta_dzf
    delta_f,delta_dxf,delta_dyf,delta_dzf=computeDelta3D(xc,yc,zc,delta_f,delta_dxf,delta_dyf,delta_dzf)
    #
    aa=np.zeros((33,33))
    for ii in range(size_box):
        for jj in range(size_box):
            newdudt,model=DerivativeSpline_v2(ii,jj,ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,spline_coeff,newtheta,newdudt,off,xstart,ystart,zstart)
            copy_model=model[0]
            copy_data=data[size_box*jj+ii]
            
            if copy_data>0:
                newerr=newerr+2*((copy_model-copy_data)-copy_data*np.log(copy_model/copy_data))
            else:
                newerr=newerr+2*copy_model
                copy_data=0
                
            aa[ii,jj]=copy_model
           
            
            t1=1-copy_data/copy_model
            
            for i1 in range(NV):
                jacobian[i1,0]+=t1*newdudt[i1,0]
                
            t2=copy_data/(copy_model**2)
            for i1 in range(NV):
                for j1 in range(NV):
                    hessian[i1*NV+j1,0]+=t2*newdudt[i1,0]*newdudt[j1,0]
                    hessian[j1*NV+i1,0]=hessian[i1*NV+j1,0]
                    
    for kk in range(iterations):
        if abs((newerr-olderr)/newerr)<tolerance:
            break
        else:
            if newerr>acceptance*olderr:
                newtheta=oldtheta
                newupdate=oldupdate
                    
                newlambda=oldlambda
                newerr=olderr
                mu=max((1+newlambda*scaleup)/(1+newlambda),1.3)
                newlambda=scaleup*newlambda
                
            elif newerr<olderr and errflag==0:
                newlambda=scaledown*newlambda
                mu=1+newlambda
                
            for i1 in range(NV):
                hessian[i1*NV+i1]=hessian[i1*NV+i1]*mu
                
            L=np.zeros((25,1))   
            U=np.zeros((25,1)) 
            errflag,L,U=choolesky(hessian,NV,L,U)
            if errflag==0:
                oldtheta=newtheta
                oldupdate=newupdate
                oldlambda=newlambda
                olderr=newerr
                
                newupdate=luEvaluate(L, U, jacobian, NV, newupdate)
                
                for i1 in range(NV):
                    if newupdate[i1,0]/oldupdate[i1,0]<-0.5:
                        maxjump[i1,0]=maxjump[i1,0]*0.5
                    newupdate[i1,0]=newupdate[i1,0]/(1+abs(newupdate[i1,0]/maxjump[i1,0]))
                    newtheta[i1,0]=newtheta[i1,0]-newupdate[i1,0]
                    
                newtheta[0,0]=max( newtheta[0,0],(size_box-1)/2-size_box/4)
                newtheta[0,0]=min( newtheta[0,0],(size_box-1)/2+size_box/4)
                
                newtheta[1,0]=max( newtheta[1,0],(size_box-1)/2-size_box/4)
                newtheta[1,0]=min( newtheta[1,0],(size_box-1)/2+size_box/4)
                
                newtheta[2,0]=max( newtheta[2,0],0)
                newtheta[2,0]=min( newtheta[2,0],spline_zsize)
                
                newtheta[4,0]=max( newtheta[4,0],0.01)
                
                newtheta[3,0]=max( newtheta[3,0],1)
                
                xc=-1*((newtheta[0,0]-size_box/2)+0.5)
                yc=-1*((newtheta[1,0]-size_box/2)+0.5)
                
                xstart=np.floor(xc)
                xc=xc-xstart
                
                ystart=np.floor(yc)
                yc=yc-ystart
                
                zstart=np.floor(newtheta[2,0])
                zc=newtheta[2,0]-zstart
                
                newerr=0
                
                jacobian=np.zeros((NV,1))
                hessian=np.zeros((NV*NV,1))
                delta_f=np.zeros((64,1))
                delta_dxf=np.zeros((64,1))
                delta_dyf=np.zeros((64,1))
                delta_dzf=np.zeros((64,1))
                delta_f,delta_dxf,delta_dyf,delta_dzf=computeDelta3D(xc, yc, zc, delta_f, delta_dxf, delta_dyf, delta_dzf)
                
                for ii in range(size_box):
                    for jj in range(size_box):
                        newdudt,model=DerivativeSpline(ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,spline_coeff,newtheta,newdudt)
                        
                        copy_model=model[0]
                        copy_data=data[size_box*jj+ii]
                        
                        if copy_data>0:
                            newerr=newerr+2*((copy_model-copy_data)-copy_data*np.log(copy_model/copy_data))
                        else:
                            newerr=newerr+2*copy_model
                            copy_data=0
                            
                        aa[ii,jj]=newerr
                       
                        
                        t1=1-copy_data/copy_model
                        
                        for i1 in range(NV):
                            jacobian[i1,0]+=t1*newdudt[i1,0]
                            
                        t2=copy_data/(copy_model**2)
                        for i1 in range(NV):
                            for j1 in range(NV):
                                hessian[i1*NV+j1,0]+=t2*newdudt[i1,0]*newdudt[j1,0]
                                hessian[j1*NV+i1,0]=hessian[i1*NV+j1,0]
            else:
                mu=max((1+newlambda*scaleup)/(1+newlambda),1.3)
                newlambda=scaleup*newlambda
        print(kk)
        print(newtheta) 
        print(newupdate) 
        print(L) 
           
                        
    return newtheta
                    
                    
                    
                    
                    
                        
                    
            
                    
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    