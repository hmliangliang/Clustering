# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\6\6 0006 19:00:00
# File:         DIFCM.py
# Software:     PyCharm
#------------------------------------
import numpy as np
import Evaluation
import math
import scipy.io as sio
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def f_object(data,U,V,m):#聚类算法的目标函数,见原文Eq.(7a)
    d = 0#初始化目标函数值
    for i in range(data.shape[0]):#每一个样本
        for j in range(V.shape[0]):#每一个聚类的类簇中心
            d = d + math.pow(U[i,j],m)*(np.linalg.norm(data[i,:]-V[j,:],2)**2)#计算目标函数值
    return d

def  DIFCM(data,k):
    '''
    本算法执行的是Wang J, Chung F, Wang S, et al. Double indices-induced FCM clustering and its integration with fuzzy subspace clustering[J].
    Pattern analysis and applications, 2014, 17(3): 549-566.
    data: 输入的数据,每一行代表一个样本
    k: 数据的类簇数目
    return: result为聚类的结果,其形式为np.array([[]])
    '''
    #配置参数,参数的具体含义见原文
    m = 2
    r = 1.1
    N_MAX = 100
    U = np.random.rand(data.shape[0],k)#初始化隶属度矩阵
    for i in range(U.shape[0]):#将隶属度矩阵转化为满足Eq.(7c)的形式
        d = 0
        for j in range(U.shape[1]):
            d = d + math.pow(U[i,j],r)
        for j in range(U.shape[1]):#归一化
            U[i,j] = math.pow(U[i,j],r)/d
        for j in range(U.shape[1]):  # 归一化
            U[i, j] = math.pow(U[i,j],1/r)
    V = data[np.random.choice(data.shape[0], k),:]#初始化随机选择k个中心点
    f_before = np.inf#上一次目标函数值
    U_before = U#上一次隶属度
    V_before = V#上一次类簇的中心点
    for i in range(N_MAX):
        if f_object(data,U,V,m) <= f_before:
            d = 0#记录Eq.(8a)的分母值
            value = 0#记录样本与隶属度相乘的值
            #固定U,更新V
            for j in range(V.shape[0]):
                for ii in range(data.shape[0]):
                    value = value + math.pow(U[ii,j],m)*data[ii,:]
                    d = d + math.pow(U[ii,j],m)#当前类簇隶属度m次方之和
                V[j,:] = value/d#按照Eq.(8a)更新中心点
            #固定V,更新U
            for ii in range(data.shape[0]):
                d = 0#记录当前样本到所有类簇的距离
                for j in range(V.shape[0]):
                    d = d + math.pow(np.linalg.norm(data[ii,:]-V[j,:],2),2*r/(m-r))
                for j in range(V.shape[0]):
                    U[ii,j] = 1/math.pow(math.pow(np.linalg.norm(data[ii,:]-V[j,:],2),2*r/(m-r))/d,1/r)
            if f_object(data,U,V,m) <= f_before:
                f_before = f_object(data, U, V, m) # 更新上一次目标函数值
                U_before = U  #更新上一次隶属度
                V_before = V  #更新上一次类簇的中心点
            else:#迭代终止
                U = U_before
                V = V_before
                break
        else:#目标函数值不再减小,迭代终止
            f_before = f_object(data, U, V, m)
            U = U_before
            V = V_before
            break
    #根据U获取类簇标签
    result = np.zeros((1,data.shape[0]))#初始化result
    for i in range(U.shape[0]):
        result[0,i] = np.argwhere(U[i,:]==max(U[i,:]))[0,0]#采用最大隶属度原则获取聚类的类标签
    return result

if __name__ == '__main__':
    start = time.time()
    name = 'wine'  # 数据集的名称
    dataFile = 'C:/Users/Administrator/Desktop/TR/datasets/' + name + '.mat'  # 数据集的路径
    data = sio.loadmat(dataFile)  # 读取数据集
    data = data[name]
    data = np.array(data)
    label = np.array([data[:, data.shape[1] - 1]])
    data = data[:, 0:data.shape[1] - 1]
    model = PCA(n_components=int(0.8*data.shape[1]))
    data = model.fit_transform(data)
    model = MinMaxScaler()
    data = model.fit_transform(data)
    k = 2  # 聚类的类簇数
    result = DIFCM(data, k)
    R, FM, K, RT, precision, recall, F1, NMI, P = Evaluation.evaluation(data, result, label)
    print('R=', R)
    print('FM=', FM)
    print('K=', K)
    print('RT=', RT)
    print('precision=', precision)
    print('recall=', recall)
    print('F1=', F1)
    print('NMI=', NMI)
    print('P=', P)
    end = time.time()
    print('算法的时间消耗为:', end - start)