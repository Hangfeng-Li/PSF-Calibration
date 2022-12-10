# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:50:51 2022

@author: lihsn
"""

import numpy as np
from numpy import matrix as mat
import math


def LM(parameter, iput,image):
    
    
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
    
    
    theta_data=image.flatten()
    
    

    # 导入数据
    Label_location = [0.9, 1.2, 2.0]#理想参数值，与后期迭代结果对比
    theta_data = [60.31857894892403, 48.25486315913922, 80.4247719318987, 80.4247719318987]#测的数据
    
    #模型数据
    lambda_data = [0.3125, 0.3125, 0.3125, 0.3125]
    xi_data = [0.0, 0.9, 2.5, 0.9]
    yi_data = [0.0, 0.0, 0.0, 0.0]
    zi_data = [2.0, 2.0, 2.0, 0.4]
    # 合并为一个矩阵，然后转置,每一行为一组λ，xi,yi,zi。
    Variable_Matrix = mat([lambda_data, xi_data, yi_data, zi_data]).T
    
    
    def Func(parameter):  # 需要拟合的函数，abc是包含三个参数的一个矩阵[[a],[b],[c]]
        x_test=parameter[0,0]
        y_test=parameter[1,0]
        z_test=parameter[2,0]
        h_test=parameter[3,0]
        b_test=parameter[4,0]
        k=int(z_test/z_step_size)
        
        grid_y_test, grid_x_test = np.mgrid[0:(input0_x_size-1)*x_pixel:200j, 0:(input0_y_size-1)*y_pixel:200j]
        
        points =  np.mgrid[0:(input0_x_size-1)*x_pixel:input0_x_size*1j, 0:(input0_y_size-1)*y_pixel:input0_y_size*1j]
        
        
        matrix_num_start=k*(input0_x_size-1)*(input0_y_size-1)
        
        img_test=np.zeros((200,200))
        
        for i in range(200):
            for j in range(200):
                # x_test=i*input0_x_size*x_pixel/100
                # y_test=j*input0_y_size*y_pixel/100
                x_test=grid_x_test[i,j]
                y_test=grid_y_test[i,j]
                matrix_num=int(x_test/x_pixel)+(int(y_test/y_pixel)*(input0_x_size-1))+matrix_num_start
                xyz=matrix_num
                if i==199:
                    i1=198
                    
                    x_test=grid_x_test[i1,j]
                    y_test=grid_y_test[i1,j]
                    matrix_num=int(x_test/x_pixel)+(int(y_test/y_pixel)*(input0_x_size-1))+matrix_num_start
                    xyz=matrix_num
                if j==199:
                    j1=198
                    
                    x_test=grid_x_test[i,j1]
                    y_test=grid_y_test[i,j1]
                    matrix_num=int(x_test/x_pixel)+(int(y_test/y_pixel)*(input0_x_size-1))+matrix_num_start
                    xyz=matrix_num
                
                if j==199 and i==199:
                    j1=198
                    i1=198
                    x_test=grid_x_test[i1,j1]
                    y_test=grid_y_test[i1,j1]
                    matrix_num=int(x_test/x_pixel)+(int(y_test/y_pixel)*(input0_x_size-1))+matrix_num_start
                    xyz=matrix_num
                    
                    
                tx0=(xyz%(input0_x_size-1))*x_pixel
                tx0_pixel=(xyz%(input0_x_size-1))
                tx1=(xyz%(input0_x_size-1)+1)*x_pixel
                tx1_pixel=(xyz%(input0_x_size-1)+1)
                ty0=(int(xyz/(input0_x_size-1))%(input0_y_size-1))*y_pixel
                ty0_pixel=(int(xyz/(input0_x_size-1))%(input0_y_size-1))
                ty1=(int(xyz/(input0_x_size-1))%(input0_y_size-1)+1)*y_pixel
                ty1_pixel=(int(xyz/(input0_x_size-1))%(input0_y_size-1)+1)
                tz0=(int(xyz/((input0_x_size-1)*(input0_y_size-1)))%(input0_z_size-1))*z_step_size
                tz0_pixel=(int(xyz/((input0_x_size-1)*(input0_y_size-1)))%(input0_z_size-1))
                tz1=(int(xyz/((input0_x_size-1)*(input0_y_size-1)))%(input0_z_size-1)+1)*z_step_size
                tz1_pixel=(int(xyz/((input0_x_size-1)*(input0_y_size-1)))%(input0_z_size-1)+1)
                for m in range(0,4):
                    for n in range(0,4):
                        for o in range(0,4):
                            
                            img_test[i,j]=matrix_parameter[matrix_num,m*16+n*4+o]*(((x_test-tx0)/(tx1-tx0))**o)*(((y_test-ty0)/(ty1-ty0))**n)*(((z_test-tz0)/(tz1-tz0))**m)+img_test[i,j]
                residual=img_test.flatten()
                return residual
    
    
    def Deriv(parameter, iput):  # 对函数求偏导
        x = parameter[0, 0]
        y = parameter[1, 0]
        z = parameter[2, 0]
        x_deriv = -4*np.pi*(iput[0, 1]-x) / (iput[0, 0] * np.sqrt(np.square(iput[0, 1]-x)+np.square(iput[0, 2]-y) + np.square(iput[0, 3]-z)))
        y_deriv = -4*np.pi*(iput[0, 2]-y) / (iput[0, 0] * np.sqrt(np.square(iput[0, 1]-x)+np.square(iput[0, 2]-y) + np.square(iput[0, 3]-z)))
        z_deriv = -4*np.pi*(iput[0, 3]-z) / (iput[0, 0] * np.sqrt(np.square(iput[0, 1]-x)+np.square(iput[0, 2]-y) + np.square(iput[0, 3]-z)))
        deriv_matrix = mat([x_deriv, y_deriv, z_deriv])
        return deriv_matrix
    
    
    n = len(theta_data)
    J = mat(np.zeros((n, 3)))  # 雅克比矩阵
    fx = mat(np.zeros((n, 1)))  # f(x)  3*1  误差
    fx_tmp = mat(np.zeros((n, 1)))
    initialization_parameters = mat([[10], [400], [30]])  # 参数初始化
    lase_mse = 0.0
    step = 0.0
    u, v = 1.0, 2.0
    conve = 100
    
    
    while conve:
        mse, mse_tmp = 0.0, 0.0
        step += 1
        for i in range(len(theta_data)):
            fx[i] = Func(initialization_parameters, Variable_Matrix[i]) - theta_data[i]  # 注意不能写成  y - Func  ,否则发散
            # print(fx[i])
            mse += fx[i, 0] ** 2
            J[i] = Deriv(initialization_parameters, Variable_Matrix[i])  # 数值求导
        mse = mse/n  # 范围约束
        H = J.T * J + u * np.eye(3)  # 3*3
        dx = -H.I * J.T * fx  # 注意这里有一个负号，和fx = Func - y的符号要对应
    
        initial_parameters_tmp = initialization_parameters.copy()
        initial_parameters_tmp = initial_parameters_tmp + dx
        for j in range(len(theta_data)):
            fx_tmp[j] = Func(initial_parameters_tmp, Variable_Matrix[j]) - theta_data[j]
            mse_tmp += fx_tmp[j, 0] ** 2
        mse_tmp = mse_tmp/n
    
        q = (mse - mse_tmp) / ((0.5 * dx.T * (u * dx - J.T * fx))[0, 0])
        print(q)
        if q > 0:
            s = 1.0 / 3.0
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
        if abs(mse - lase_mse) < math.pow(0.1, 14):
            break
        lase_mse = mse  # 记录上一个 mse 的位置
        conve -= 1
    print(lase_mse)
    print(initialization_parameters)