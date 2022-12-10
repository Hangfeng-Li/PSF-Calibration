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
    
    theta_data=image.flatten()
    
    
    init_parameter=mat([x_test, y_test, z_test, h_test,b_test]).T
    
    #####################
    
    x_test=0
    y_test=0
    z_test=0.4
    h_test=90
    b_test=3
    
    input0_x_size=33
    input0_y_size=33
    input0_z_size=22
    
    z_step_size=0.1
    x_pixel=0.065
    y_pixel=0.065
    
    
    
    big_win_size_small=73
    win_size_small=33
    
    
    theta_data=image.flatten()
    

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
    
    
    def Func(init_parameter):  # 需要拟合的函数，abc是包含三个参数的一个矩阵[[a],[b],[c]]
        x_test_c=init_parameter[0,0]
        y_test_c=init_parameter[1,0]
        z_test_c=init_parameter[2,0]
        h_test_c=init_parameter[3,0]
        b_test_c=init_parameter[4,0]
        img_test_c=generate_cail_psf(z_test_c,y_test_c,x_test_c,h_test_c,b_test_c,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)
       
        residual=img_test_c.flatten()
        return residual
    
    
    def Deriv(init_parameter):  # 对函数求偏导
        x_test_c=init_parameter[0,0]
        y_test_c=init_parameter[1,0]
        z_test_c=init_parameter[2,0]
        h_test_c=init_parameter[3,0]
        b_test_c=init_parameter[4,0]
        
        h_deriv,x_deriv,y_deriv,z_deriv=dev_cail_psf(z_test_c,y_test_c,x_test_c,h_test_c,b_test_c,big_win_size_small,win_size_small,x_pixel,y_pixel,z_step_size,cail_psf_size,matrix_parameter)
        h_deriv=h_deriv.flatten()
        x_deriv=x_deriv.flatten()
        y_deriv=y_deriv.flatten()
        z_deriv=z_deriv.flatten()
        b_deriv=np.ones((win_size_small*win_size_small,1))
        deriv_matrix = mat([h_deriv, x_deriv, y_deriv,z_deriv,b_deriv])
        return deriv_matrix
    
    
    n = len(theta_data)
    J = mat(np.zeros((n, 5)))  # 雅克比矩阵
    fx = mat(np.zeros((n, 1)))  # f(x)  5*1  误差
    fx_tmp = mat(np.zeros((n, 1)))
    initialization_parameters = mat([h_test, x_test, y_test,z_test,b_test]).T  # 参数初始化
    lase_mse = 0.0
    step = 0.0
    u, v = 1.0, 2.0
    conve = 20
    
    
    while conve:
        mse, mse_tmp = 0.0, 0.0
        step += 1
        
        fx=Func(initialization_parameters) - theta_data
        mse=np.sum(fx**2)
        J=Deriv(initialization_parameters) 
        
        mse = mse/n  # 范围约束
        H = J.T * J + u * np.eye(5)  # 5*5
        dx = -H.I * J.T * fx  # 注意这里有一个负号，和fx = Func - y的符号要对应
    
        initial_parameters_tmp = initialization_parameters.copy()
        initial_parameters_tmp = initial_parameters_tmp + dx
        
        fx_tmp = Func(initial_parameters_tmp) - theta_data
        mse_tmp =np.sum(fx_tmp**2)
        mse_tmp = mse_tmp/n
    
        q = (mse - mse_tmp) / ((0.5 * dx.T * (u * dx - J.T * fx))[0, 0])
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
        if abs(mse - lase_mse) < math.pow(0.1, 14):
            break
        lase_mse = mse  # 记录上一个 mse 的位置
        conve -= 1
    print(lase_mse)
    print(initialization_parameters)