# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:12:18 2022

@author: lihsn
"""

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
from computeDelta3Dj_v2 import computeDelta3Dj_v2
from numpy import matrix as mat

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
iterations=20

# cor=np.column_stack(cor)
init_parameters=[13.5,13.8,16,0.01,0.009]#13,13,16,1,0.01

init_parameters=np.column_stack(init_parameters).T

# spline_coeff=s_coeff.flatten()
spline_coeff=s_coeff





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
    maxjump=[1,1,1,100,20]
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
    
    # xc=-1*(newtheta[0]-size_box/2+0.5)
    # yc=-1*(newtheta[1]-size_box/2+0.5)
    
    # off=np.floor((spline_xsize+1-size_box)/2)
    
    # xstart=np.floor(xc)
    # xc=xc-xstart
    
    # ystart=np.floor(yc)
    # yc=yc-ystart
    
    # zstart=np.floor(newtheta[2])
    # zc=newtheta[2]-zstart
    
    
    off = np.floor(((spline_xsize+1)-Npixels)/2)
    
    
    
    xcenter = newtheta[0]
    ycenter = newtheta[1]
    zcenter = newtheta[2]
    
    xc = -1*(xcenter - Npixels/2+0.5)
    yc = -1*(ycenter - Npixels/2+0.5)
    zc = zcenter - math.floor(zcenter)
    
    xstart = math.floor(xc)
    xc = xc - xstart

    ystart = math.floor(yc)
    yc = yc - ystart


    zstart = math.floor(zcenter)
    
    
    
    
    newerr=0
    jacobian=np.zeros((NV,1))
    hessian=np.zeros((NV*NV,1))
    
    ####带入xc，yc，zc等计算delta_f,delta_dxf,delta_dyf,delta_dzf
    delta_f,delta_dxf,delta_ddxf,delta_dyf,delta_ddyf,delta_dzf,delta_ddzf=computeDelta3Dj_v2(xc,yc,zc)
    #
    aa=np.zeros((33,33))
    bb=np.zeros((33,33))
    for ii in range(size_box):
        for jj in range(size_box):
            newdudt,model=DerivativeSpline_v2(ii,jj,ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,spline_coeff,newtheta,newdudt,off,xstart,ystart,zstart)
            copy_model=model[0]
            copy_data=data[size_box*ii+jj]
            
            if copy_data>0:
                newerr=newerr+2*((copy_model-copy_data)-copy_data*np.log(copy_model/copy_data))
            else:
                newerr=newerr+2*copy_model
                copy_data=0
                
            aa[ii,jj]=copy_data
            bb[ii,jj]=copy_model
           
            
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
            errflag=choolesky(hessian,NV,L,U)
            if errflag==0:
                oldtheta=newtheta
                oldupdate=newupdate
                oldlambda=newlambda
                olderr=newerr
                
                # hessian=mat(hessian.reshape([5,5]))
                # newupdate = np.dot(hessian.I , jacobian)*newerr
                
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
                
                xcenter = newtheta[0]
                ycenter = newtheta[1]
                zcenter = newtheta[2]
                
                xc = -1*(xcenter - Npixels/2+0.5)
                yc = -1*(ycenter - Npixels/2+0.5)
                zc = zcenter - np.floor(zcenter)
                
                xstart = np.floor(xc)
                xc = xc - xstart

                ystart = np.floor(yc)
                yc = yc - ystart


                zstart =np.floor(zcenter)
                
                newerr=0
                
                jacobian=np.zeros((NV,1))
                hessian=np.zeros((NV*NV,1))
                delta_f=np.zeros((64,1))
                delta_dxf=np.zeros((64,1))
                delta_dyf=np.zeros((64,1))
                delta_dzf=np.zeros((64,1))
                delta_f,delta_dxf,delta_ddxf,delta_dyf,delta_ddyf,delta_dzf,delta_ddzf=computeDelta3Dj_v2(xc,yc,zc)
                
                for ii in range(size_box):
                    for jj in range(size_box):
                        newdudt,model=DerivativeSpline_v2(ii,jj,ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,spline_coeff,newtheta,newdudt,off,xstart,ystart,zstart)
                        
                        copy_model=model[0]
                        copy_data=data[size_box*ii+jj]
                        
                        if copy_data>0:
                            newerr=newerr+2*((copy_model-copy_data)-copy_data*np.log(copy_model/copy_data))
                        else:
                            newerr=newerr+2*copy_model
                            aa[ii,jj]=copy_data
                            bb[ii,jj]=copy_model
                            
                        aa[ii,jj]=copy_model
                       
                        
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
                    
                    
                    
plt.imshow(simpsf,origin='lower')    
plt.imshow(aa,origin='lower')  
plt.imshow(bb,origin='lower')                       
                    
            
                    
            
     n = len(data)
     J = mat(np.zeros((n, 5)))  # 雅克比矩阵
     fx = mat(np.zeros((n, 1)))  # f(x)  5*1  误差
     fx_tmp = mat(np.zeros((n, 1)))
     initialization_parameters = mat([ x_test, y_test,z_test,h_test,b_test]).T  # 参数初始化
     lase_mse = 0.0
     step = 0.0
     u, v = 1.0, 2.0
     conve = 100
     
     
     while conve:
         mse, mse_tmp = 0.0, 0.0
         step += 1
         for ii in range(size_box):
             for jj in range(size_box):
                 newdudt,model[ii,jj]=DerivativeSpline_v2(ii,jj,ii+xstart+off,jj+ystart+off,zstart,spline_xsize,spline_ysize,spline_zsize,delta_f,delta_dxf,delta_dyf,delta_dzf,spline_coeff,newtheta,newdudt,off,xstart,ystart,zstart)
         fx=model.flatten() -data
         mse=np.sum(fx**2)
         J=Deriv(initialization_parameters,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter) 
         
         mse = mse/n  # 范围约束
         H = mat(np.dot(J.T,J )+ u * np.eye(5) ) # 5*5
         dx = -H.I * J.T * fx  # 注意这里有一个负号，和fx = Func - y的符号要对应
     
         initial_parameters_tmp = initialization_parameters.copy()
         initial_parameters_tmp = initial_parameters_tmp + dx
         
         fx_tmp = Func(initial_parameters_tmp,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter) - theta_data
         mse_tmp =np.sum(fx_tmp**2)
         mse_tmp = mse_tmp/n
     
         q = (mse - mse_tmp) / ((0.5 * dx.T * (u * dx - np.dot(J.T , fx)))[0, 0])
         print(q)
         if q > 0:
             s = 1.0 / 5.0
             v = 2
             mse = mse_tmp
             initialization_parameters = initial_parameters_tmp
             temp = 1 - pow(2 * q - 1, 3)
             if s > temp:
                 u = u * s
             else:
                 u = u * temp
         else:
             u = u * v
             v = 2 * v
             mse = mse_tmp
             initialization_parameters = initial_parameters_tmp
         print("step = %d,parameters(mse-lase_mse) = " % step, abs(mse - lase_mse))
         if abs(mse - lase_mse) < math.pow(0.1, 50):
             break
         lase_mse = mse  # 记录上一个 mse 的位置
         conve -= 1
         print(lase_mse)
         print(initialization_parameters)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    