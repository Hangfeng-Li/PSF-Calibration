# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:10:43 2022

@author: e0947330
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 21:18:46 2022

@author: lihsn
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:06:20 2022

@author: e0947330
"""

import numpy as np
from numpy import matrix as mat
import math
from generate_cail_psf_spline import generate_cail_psf
from dev_cail_psf_spline import dev_cail_psf

def LM(parameter,image):
    
    
    x_test=parameter[0,0]
    y_test=parameter[1,0]
    z_test=parameter[2,0]
    h_test=parameter[3,0]
    b_test=parameter[4,0]
    
    input0_x_size=parameter[5,0]
    input0_y_size=parameter[6,0]
    input0_z_size=parameter[7,0]
    
    z_step_size=parameter[8,0]
    x_pixel=parameter[9,0]
    y_pixel=parameter[10,0]
    
    matrix_parameter=parameter[11,0]
    
    big_win_size_small=parameter[12,0]
    win_size_small=parameter[13,0]
    cail_psf_size=parameter[14,0]
    
   
    theta_data=image.reshape(-1, 1)
    
    init_parameter=mat([x_test, y_test, z_test, h_test,b_test]).T
    
    #####################
    
    x_test=-0.3
    y_test=0.4
    z_test=0.4
    h_test=135
    b_test=20
    
    
    input0_x_size=33
    input0_y_size=33
    input0_z_size=22
    
    z_step_size=0.1
    x_pixel=0.065
    y_pixel=0.065
    
    
    
    big_win_size_small=73
    win_size_small=33
    
    cail_psf_size=cail_psf.shape
    
    

    # # 导入数据
    # Label_location = [0.9, 1.2, 2.0]#理想参数值，与后期迭代结果对比
    # theta_data = [60.31857894892403, 48.25486315913922, 80.4247719318987, 80.4247719318987]#测的数据
    
    # #模型数据
    # lambda_data = [0.3125, 0.3125, 0.3125, 0.3125]
    # xi_data = [0.0, 0.9, 2.5, 0.9]
    # yi_data = [0.0, 0.0, 0.0, 0.0]
    # zi_data = [2.0, 2.0, 2.0, 0.4]
    # # 合并为一个矩阵，然后转置,每一行为一组λ，xi,yi,zi。
    # Variable_Matrix = mat([lambda_data, xi_data, yi_data, zi_data]).T
    
    
    def Func(init_parameter,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter):  # 需要拟合的函数，abc是包含三个参数的一个矩阵[[a],[b],[c]]
        x_test_c=init_parameter[0,0]
        y_test_c=init_parameter[1,0]
        z_test_c=init_parameter[2,0]
        h_test_c=init_parameter[3,0]
        b_test_c=init_parameter[4,0]
        img_test_c=generate_cail_psf(z_test_c,y_test_c,x_test_c,h_test_c,b_test_c,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)
       
        residual=img_test_c.reshape(-1, 1)
        return residual
    
    

    
    theta_data_min=np.min(theta_data)
    theta_data=theta_data-theta_data_min+1
    
    
    
   
    def xmle0(residual,theta_data):
        
        fi_sum=2*np.sum(residual-theta_data)
       
        secondary=2*np.sum(theta_data*np.log(np.abs(residual/theta_data)))
        xmle=fi_sum-secondary
        return xmle
    
    
    dev_b_img_test2=np.ones((33,33))
    
    
    
    def dxmle0(dev_x_img_test2,dev_y_img_test2,dev_z_img_test2,dev_h_img_test2,dev_b_img_test2,residual,theta_data):
        
        dxmle_x=np.sum((1-theta_data/residual)*dev_x_img_test2.reshape(-1, 1))
        dxmle_y=np.sum((1-theta_data/residual)*dev_y_img_test2.reshape(-1, 1))
        dxmle_z=np.sum((1-theta_data/residual)*dev_z_img_test2.reshape(-1, 1))
        dxmle_h=np.sum((1-theta_data/residual)*dev_h_img_test2.reshape(-1, 1))
        dxmle_b=np.sum((1-theta_data/residual)*dev_b_img_test2.reshape(-1, 1))
        deriv_matrix = [ dxmle_x, dxmle_y,dxmle_z,dxmle_h,dxmle_b.reshape(-1, 1)]
        deriv_matrix=-np.column_stack(deriv_matrix).T
        
        
        return deriv_matrix
    
    
        
    def ddxmle(deriv_matrix,residual,theta_data):
        
        a_ij=np.zeros((5,5))
        for i1 in range(5):
            for j1 in range(5):
                a_ij[i1,j1]=np.sum(deriv_matrix[i1]*deriv_matrix[j1]*(theta_data/(residual**2)))
        return a_ij
        
    
    lamda=0.1
    maxthe=-0.1*np.ones((5,1))
    sign=np.ones((5,1))
    conve=50
    step=0
    C=[1,1,1,1000,100]
    C=np.column_stack(C).T
    initialization_parameters = mat([ x_test, y_test,z_test,h_test,b_test]).T  # 参数初始化
    flag1=0
    
    while conve:
        
         if step==0:
             initial_parameters = initialization_parameters
         
         residual=Func(initial_parameters,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)
         residual_min=np.min(residual)
         residual=residual-residual_min+1
         
         xmle=xmle0(residual,theta_data)
         dev_x_img_test2,dev_y_img_test2,dev_z_img_test2,dev_h_img_test2=dev_cail_psf(initial_parameters[2,0],initial_parameters[1,0],initial_parameters[0,0],initial_parameters[3,0],initial_parameters[4,0],big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)
         dxmle=dxmle0(dev_x_img_test2,dev_y_img_test2,dev_z_img_test2,dev_h_img_test2,dev_b_img_test2,residual,theta_data)
         a_ij=ddxmle(dxmle,residual,theta_data)
         
         H_lamda=mat(a_ij+lamda * np.eye(5) )
         
         dx_init=H_lamda.I*dxmle
         dx_final_new=dx_init/(1+np.abs(dx_init/maxthe))
         
         initial_parameters_tmp=initial_parameters +dx_final_new
         
         if step>0:
             
             sign=np.multiply(dx_final_old,dx_final_new)
             for i1 in range(sign.shape[0]):
                 if sign[i1,0]<0:
                     maxthe[i1,0]=maxthe[i1,0]*0.5
         dx_final_old=dx_final_new
         residual_tmp=Func(initial_parameters_tmp,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)
         residual_tmp_min=np.min(residual_tmp)
         residual_tmp=residual_tmp-residual_tmp_min+1
         
         xmle_tmp=xmle0(residual_tmp,theta_data)
         
         if xmle_tmp<xmle:
             lamda=lamda/10
             initial_parameters=initial_parameters_tmp
         elif xmle_tmp>=1.5*xmle:
             lamda=lamda*10
             initial_parameters=initial_parameters
         else:
             lamda=lamda
             initial_parameters=initial_parameters_tmp
         
         if abs((xmle_tmp-xmle)/xmle)<1e-6:
             break

         step+=1
         
         conve -= 1
         print(step)
         print(initial_parameters)
         print(dx_final_new)
         print(abs(xmle_tmp/xmle))
         print(lamda)
         print(maxthe)
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         # A=H_lamda+H_lamda.T
         # eig_A=np.linalg.eigvals(A)
         if np.all(eig_H<=0):
             lamda=lamda*10
         else:
             dx=H_lamda.I*dxmle
             initial_parameters_tmp=initial_parameters+dx
             residual_tmp=Func(initial_parameters_tmp,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)
             residual_tmp_min=np.min(residual_tmp)
             residual_tmp=residual_tmp-residual_tmp_min+1
             xmle_tmp=xmle0(residual_tmp,theta_data)
             flag1=1
             if xmle_tmp<xmle:
                 lamda=lamda*10
             else:
                initial_parameters=initial_parameters_tmp
                lamda=lamda/10
         if flag1==1 and   abs(xmle_tmp-xmle)<0.01**15:
             break
         conve -= 1
         print(step)
         print(initial_parameters)
         print(dx)
         print(abs(xmle_tmp-xmle))
         print(lamda)
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         deriv_matrix=Deriv(initial_parameters,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)
         a_ij=ddxmle(deriv_matrix,residual,theta_data)
         B=dxmle(deriv_matrix,residual,theta_data)
         
         # U=np.dot(np.linalg.pinv(a_ij),B)
         
         xmle=xmle0(residual,theta_data)
         
         # dx=-U/(1+np.abs(U)/C)
         
         a_ij_lamda=mat(a_ij*(1+lamda * np.eye(5)))
         dx=np.dot(np.linalg.pinv(a_ij_lamda),B)
         initial_parameters_tmp= initial_parameters+dx
         
         if initial_parameters_tmp[0,0]<-0.5 :
             initial_parameters_tmp[0,0]=-0.5
         if initial_parameters_tmp[0,0]>0.5 :
             initial_parameters_tmp[0,0]=0.5
         if initial_parameters_tmp[1,0]<-0.5 :
             initial_parameters_tmp[1,0]=-0.5
         if initial_parameters_tmp[1,0]>0.5 :
             initial_parameters_tmp[1,0]=0.5
         if initial_parameters_tmp[2,0]<0 :
             initial_parameters_tmp[2,0]=0
         if initial_parameters_tmp[2,0]>1.5 :
             initial_parameters_tmp[2,0]=1.5      
             
         residual_tmp=Func(initial_parameters_tmp,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)
         xmle_tmp=xmle0(residual_tmp,theta_data)
         
         if np.abs(xmle_tmp-xmle)<math.pow(0.1, 14):
             break
         
         if xmle_tmp>xmle:
             lamda=lamda*10
         elif xmle_tmp<xmle:
             lamda=lamda/10
             initial_parameters=initial_parameters_tmp
             xmle=xmle_tmp
         else:
             break
         
         
         
         lase_mse = xmle  # 记录上一个 mse 的位置
         conve -= 1
         print(step)
         print(lase_mse)
         print(xmle_tmp)
         print(initial_parameters)
         print(dx)
         
     
     
     
    
    
    
    
    
    
    
    lamda=0.1
    updateJ=1
    n = len(theta_data)
    J = mat(np.zeros((n, 5)))  # 雅克比矩阵
    fx = mat(np.zeros((n, 1)))  # f(x)  5*1  误差
    fx_tmp = mat(np.zeros((n, 1)))
    initialization_parameters = mat([ x_test, y_test,z_test,h_test,b_test]).T  # 参数初始化
    lase_mse = 0.0
    step = 0.0
    
    conve = 10
    
    
    while conve:
        
        
        if step==0:
           initial_parameters = initialization_parameters 
        step += 1
        
        if updateJ==1:
            
            fx=Func(initial_parameters,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter) - theta_data
            mse=np.sum(fx**2)
            J=Deriv(initial_parameters,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter) 
            mse = mse/n  # 范围约束
            H = np.dot(J.T,J )   # 3*3
            
        
        H_tmp=mat(H + lamda * np.eye(5))
        dx = -H_tmp.I * J.T * fx  # 注意这里有一个负号，和fx = Func - y的符号要对应
        initial_parameters_tmp = initial_parameters + dx
        
        fx_tmp = Func(initial_parameters_tmp,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter) - theta_data
        mse_tmp =np.sum(fx_tmp**2)
        mse_tmp = mse_tmp/n
        
        if mse_tmp< mse:
            lamda=lamda/10
            initial_parameters = initial_parameters_tmp
            mse = mse_tmp
            updateJ=1
        else:
            updateJ=0
            lamda=lamda*10
            
           
        # print("step = %d,parameters(mse-lase_mse) = " % step, abs(mse - lase_mse))
        # if abs(mse - lase_mse) < math.pow(0.1, 14):
        #     break
        lase_mse = mse  # 记录上一个 mse 的位置
        conve -= 1
        print(lase_mse)
        print(initial_parameters)
        
        
        
        
        
        
        step += 1
        
        fx=Func(initialization_parameters,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter) - theta_data
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
            temp = 1 - pow(2 * q - 1, 5)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    