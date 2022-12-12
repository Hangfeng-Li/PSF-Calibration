# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 23:11:29 2022

@author: lihsn
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
    
    x_test=-0.6
    y_test=0.4
    z_test=0.8
    h_test=147
    b_test=6
    
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
    
    theta_data=theta_data+1
    def L(residual,theta_data,cail_psf_size):
        I_exp_multiply=0
        L_F_I=1
        for i1 in range(33*33):
            I_exp_multiply=I_exp_multiply+theta_data[i1]
            L_F_I=((np.abs(residual[i1]))**(theta_data[i1]))/(I_exp_multiply)*np.exp(-np.abs(residual[i1]))*L_F_I
        return L_F_I
    
    def dL(residual,theta_data,cail_psf_size,init_parameter):
        h_x=0.01
        h_y=0.01
        h_z=0.01
        h_h=1
        h_b=1
        h_matrix = [h_x, h_y, h_z,h_h,h_b]
        h_matrix=np.column_stack(h_matrix).T
        init_parameter_xz=init_parameter
        init_parameter_yz=init_parameter
        init_parameter_zz=init_parameter
        init_parameter_hz=init_parameter
        init_parameter_bz=init_parameter
        
        init_parameter_xf=init_parameter
        init_parameter_yf=init_parameter
        init_parameter_zf=init_parameter
        init_parameter_hf=init_parameter
        init_parameter_bf=init_parameter
        
        init_parameter_xz[0,0]=h_x+init_parameter_xz[0,0]
        init_parameter_yz[1,0]=h_y+init_parameter_yz[1,0]
        init_parameter_zz[2,0]=h_z+init_parameter_zz[2,0]
        init_parameter_hz[3,0]=h_h+init_parameter_hz[3,0]
        init_parameter_bz[4,0]=h_b+init_parameter_bz[4,0]
        
        init_parameter_xf[0,0]=-h_x+init_parameter_xf[0,0]
        init_parameter_yf[1,0]=-h_y+init_parameter_yf[1,0]
        init_parameter_zf[2,0]=-h_z+init_parameter_zf[2,0]
        init_parameter_hf[3,0]=-h_h+init_parameter_hf[3,0]
        init_parameter_bf[4,0]=-h_b+init_parameter_bf[4,0]
        
        
        dL_x=(Func(init_parameter_xz,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)-Func(init_parameter_xf,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter))/(2*h_x)
        dL_y=(Func(init_parameter_yz,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)-Func(init_parameter_yf,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter))/(2*h_y)
        dL_z=(Func(init_parameter_zz,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)-Func(init_parameter_zf,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter))/(2*h_z)
        dL_h=(Func(init_parameter_hz,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)-Func(init_parameter_hf,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter))/(2*h_h)
        dL_b=(Func(init_parameter_bz,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)-Func(init_parameter_bf,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter))/(2*h_b)
        
        ddL_x=(Func(init_parameter_xz,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)+Func(init_parameter_xf,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)-2*residual)/(h_x**2)
        ddL_y=(Func(init_parameter_yz,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)+Func(init_parameter_yf,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)-2*residual)/(h_y**2)
        ddL_z=(Func(init_parameter_zz,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)+Func(init_parameter_zf,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)-2*residual)/(h_z**2)
        ddL_h=(Func(init_parameter_hz,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)+Func(init_parameter_hf,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)-2*residual)/(h_h**2)
        ddL_b=(Func(init_parameter_bz,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)+Func(init_parameter_bf,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)-2*residual)/(h_b**2)
    
    
    
        dL_o = [dL_x, dL_y, dL_z,dL_h,dL_b]
        ddL_o = [ddL_x, ddL_y, ddL_z,ddL_h,ddL_b]
        
        dL_o_matrix=np.column_stack(dL_o)
        ddL_o_matrix=np.column_stack(ddL_o)
        return dL_o_matrix,ddL_o_matrix
    
    def Deriv(init_parameter,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter):  # 对函数求偏导
        x_test_c=init_parameter[0,0]
        y_test_c=init_parameter[1,0]
        z_test_c=init_parameter[2,0]
        h_test_c=init_parameter[3,0]
        b_test_c=init_parameter[4,0]
        
        h_deriv,x_deriv,y_deriv,z_deriv=dev_cail_psf(z_test_c,y_test_c,x_test_c,h_test_c,b_test_c,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)
        h_deriv=h_deriv.reshape(-1, 1)
        x_deriv=x_deriv.reshape(-1, 1)
        y_deriv=y_deriv.reshape(-1, 1)
        z_deriv=z_deriv.reshape(-1, 1)
        b_deriv=np.ones((win_size_small*win_size_small,1))
        deriv_matrix = [h_deriv, x_deriv, y_deriv,z_deriv,b_deriv]
        deriv_matrix=np.column_stack(deriv_matrix)
        return deriv_matrix
    
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
    old_L_F_I=0
    L_F_I=1
    
    while conve:
        
        
        if step==0:
           initial_parameters = initialization_parameters 
        step += 1
        residual=Func(initial_parameters,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter) - theta_data
        if abs(old_L_F_I-L_F_I)<0.1**15:
            break
        
        L_F_I=L(residual,theta_data,cail_psf_size) 
        erro=old_L_F_I-L_F_I
        dL_o_matrix,ddL_o_matrix=dL(residual,theta_data,cail_psf_size,init_parameter)
        
        initial_parameters=initial_parameters-lamda*(1/ddL_o_matrix)*dL_o_matrix    
        old_L_F_I=L_F_I
        
        print(step)
        print(L_F_I)
        print(initial_parameters)
        print(abs(erro))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    