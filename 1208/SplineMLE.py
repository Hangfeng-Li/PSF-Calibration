# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 23:11:14 2022

@author: lihsn
"""

import numpy as np


def SplineMLE(box, data,spline_coeff,spline_xsize,spline_ysize,spline_zsize,size_box,iterations,init_parameters):
    
    NV=5
    M=np.zeros((NV*NV,1))
    Diag=np.zeros((NV,1))
    Minv=np.zeros((NV*NV))
    
    xstart=init_parameters[0]
    ystart=init_parameters[1]
    zstart=init_parameters[2]
    Nstart=init_parameters[3]
    Bgstart=init_parameters[4]
    
    
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
    
    Nmax=np.max(data)
    newtheta=init_parameters
    newtheta[4]=np.max([newtheta[4],0.01])
    newtheta[3]=(Nmax-newtheta[3])/spline_coeff[(spline_zsize/2)*(spline_xsize*spline_ysize)+(spline_ysize/2)*spline_xsize+(spline_xsize/2)]*4
    #(x,y,bg,I,z)
    maxjump[4]=np.max([newtheta[4],maxjump[4]])
    maxjump[3]=np.max([newtheta[3],maxjump[3]])
    maxjump[2]=np.max([spline_zsize/3,maxjump[2]])
    
    oldtheta=newtheta
    
    xc=-1*(newtheta[0]-size_box/2+0.5)
    xc=-1*(newtheta[1]-size_box/2+0.5)
    
    off=(spline_xsize+1-size_box)/2
    
    xstart=xc
    xc=xc-xstart
    
    ystart=yc
    yc=yc-ystart
    
    zstart=newtheta[2]
    zc=newtheta[2]-zstart
    
    newerr=0
    
    ####带入xc，yc，zc等计算delta_f,delta_dxf,delta_dyf,delta_dzf
    delta_f,delta_dxf,delta_dyf,delta_dzf=computeDelta3D(xc,yc,zc,delta_f,delta_dxf,delta_dyf,delta_dzf)
    #
    
    for ii in range(size_box):
        for jj in range(size_box):
            newdudt,model=DerivativeSpline(ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,spline_coeff,newtheta,newdudt)
            
            copy_data=data
            copy_data[copy_data<0]=0
            data_zheng=data
            data_zheng[data_zheng<=0]=0.1
            
            newerr=np.sum(2*((model-copy_data)-copy_data*np.log(model/data_zheng)))
            
            t1=1-data/model
            
            for i1 in range(NV):
                jacobian[i1]+=t1*newdudt[i1]
                
            t2=data/(model**2)
            for i1 in range(NV):
                for j1 in range(NV):
                    hessian[i1*NV+j1]+=t2*newdudt[i1]*newdudt[j1]
                    hessian[j1*NV+i1]=hessian[i1*NV+j1]
                    
    for kk in range(iterations):
        if abs((newerr-olderr)/newerr)<tolerance:
            break
        else:
            if newerr>acceptance*olderr:
                newtheta=oldtheta
                newupdate=oldupdate
                    
                newlambda=oldlambda
                newerr=olderr
                mu=np.max((1+newlambda*scaleup)/(1+newlambda),1.3)
                newlambda=scaleup*newlambda
                
            elif newerr<olderr and errflag==0:
                newlambda=scaledown*newlambda
                mu=1+newlambda
                
            for i1 in range(NV):
                hessian[i1*NV+i1]=hessian[i1*NV+i1]*mu
                errflag,L,U=cholesky(hessian,NV,L,U)
            if errflag==0:
                newtheta=oldtheta
                newupdate=oldupdate
                oldlambda=newlambda
                olderr=newerr
                newupdate=luEvaluate(L, U, jacobian, NV, newupdate)
                
                for i1 in range(NV):
                    if newupdate[i1]/oldupdate[i1]<-0.5:
                        maxjump[i1]=maxjump[i1]*0.5
                    newupdate[i1]=newupdate[i1]/(1+abs(newupdate[i1]/maxjump[i1]))
                    newtheta[i1]=newtheta[i1]-newupdate[i1]
                    
                newtheta[0]=np.max( newtheta[0],(size_box-1)/2-size_box/4)
                newtheta[0]=np.min( newtheta[0],(size_box-1)/2-size_box/4)
                
                newtheta[1]=np.max( newtheta[1],(size_box-1)/2-size_box/4)
                newtheta[1]=np.min( newtheta[1],(size_box-1)/2-size_box/4)
                
                newtheta[2]=np.max( newtheta[2],0)
                newtheta[2]=np.min( newtheta[2],spline_zsize)
                
                newtheta[3]=np.max( newtheta[3],0.01)
                
                newtheta[4]=np.max( newtheta[4],1)
                
                xc=-1*((newtheta[0]-size_box/2)+0.5)
                yc=-1*((newtheta[1]-size_box/2)+0.5)
                
                xstart=xc
                xc=xc-xstart
                
                ystart=yc
                yc=yc-ystart
                
                zstart=newtheta[2]
                zc=newtheta[2]-zstart
                
                newerr=0
                
                delta_f,delta_dxf,delta_dyf,delta_dzf=computeDelta3D(xc, yc, zc, delta_f, delta_dxf, delta_dyf, delta_dzf)
                
                for ii in range(size_box):
                    for jj in range(size_box):
                        newdudt,model=DerivativeSpline(ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,spline_coeff,newtheta,newdudt,model)
                        
                        copy_data=data
                        copy_data[copy_data<0]=0
                        data_zheng=data
                        data_zheng[data_zheng<=0]=0.1
                        
                        newerr=np.sum(2*((model-copy_data)-copy_data*np.log(model/data_zheng)))
                        
                        t1=1-data/model
                        
                        for i1 in range(NV):
                            jacobian[i1]+=t1*newdudt[i1]
                            
                        t2=data/(model**2)
                        for i1 in range(NV):
                            for j1 in range(NV):
                                hessian[i1*NV+j1]+=t2*newdudt[i1]*newdudt[j1]
                                hessian[j1*NV+i1]=hessian[i1*NV+j1]
            else:
                mu=np.max((1+newlambda*scaleup)/(1+newlambda),1.3)
                newlambda=scaleup*newlambda
                        
                        
    return newtheta
                    
                    
                    
                    
                    
                        
                    
            
                    
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    