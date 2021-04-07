# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\6\4 0004 09:48:48
# File:         SEANC.py
# Software:     PyCharm
#------------------------------------

import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.io as sio
import Evaluation
from sklearn.preprocessing import MinMaxScaler

def f_function(F,S):#目标函数的值按照Eq.(17)式计算
    value = 0
    for i in range(F.shape[0]):
        for j in range(F.shape[0]):
            value = value + np.linalg.norm(F[i,:]-F[j,:],2)
    return value

def SEANC(data,k):
    '''本算法执行的是Wang Q, Qin Z, Nie F, et al. Spectral embedded adaptive neighbors clustering[J]. IEEE transactions on
    neural networks and learning systems, 2018,30(4): 1265-1271.
    输入参数  data:为一个n*d的一个矩阵        k:聚类的类簇数目  delta,lamda,u,a,b为参数'''
    data = np.array(data)
    X = data.transpose()
    delta = data.shape[1]
    lamda = 0.2
    u = 0.2
    alpha = 50
    beta = 0.002
    N_MAX = 200#设置最大的运行次数
    W = np.zeros((data.shape[0],data.shape[0]))
    D = np.zeros((data.shape[0],data.shape[0]))
    for i in range(data.shape[0]):#计算W矩阵和D矩阵
        for j in range(data.shape[0]):
            W[i,j] = math.exp(-np.linalg.norm(data[i,:]-data[j,:])*np.linalg.norm(data[i,:]-data[j,:])/(2*delta*delta))
        aver = sum(W[i,:])/data.shape[0]
        for j in range(data.shape[0]):
            if W[i,j] < 0.8*aver:
                W[i,j] = 0
                W[j,i] = 0
        D[i,i] = sum(W[i,:])
    L = np.eye(data.shape[0]) - np.dot(np.linalg.inv(D),W)#计算拉普拉斯矩阵按照Eq.(3)
    Hn = np.eye(data.shape[0]) -1/data.shape[0] * np.ones(data.shape[0])#计算中心矩阵
    A= L + lamda*(Hn-np.dot(X.transpose(),np.dot(np.linalg.inv(np.dot(X,X.transpose())+u*np.eye(data.shape[1])),X)))
    eigenvalue,F = np.linalg.eig(A) #按照算法1中第2步计算特征值和特征向量
    seq = np.argsort(eigenvalue)
    seq = seq[1:k+1]
    F = F[:,seq]
    S = np.random.rand(data.shape[0],data.shape[0])
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            S[i,j] = S[i,j]/sum(S[i,:])
    F_before = F#记录上一次F的值
    S_before = S#记录上一次S的值
    f_before = f_function(F,S)#初始化目标函数值
    for i in range(N_MAX):
        if f_function(F,S) <= f_before:#目标函数值下降
            DF = np.zeros((data.shape[0],data.shape[0]))
            for ii in range(data.shape[0]):
                DF[ii,ii] = sum(S[ii,:])#计算权值对角矩阵
            LF = np.eye(data.shape[0]) - np.dot(np.linalg.inv(DF),S)
            #更新G矩阵
            eigenvalue, G = np.linalg.eig(LF)
            seq = np.argsort(eigenvalue)
            seq = seq[1:k+1]
            G = G[:,seq]
            #更新参数S
            for ii in range(S.shape[0]):
                for jj in range(S.shape[1]):
                    dij = np.linalg.norm(F[ii,:]-F[jj,:],2)**2 + beta*np.linalg.norm(G[ii,:]-G[jj,:],2)**2
                    S[ii,jj] = -dij/(2*alpha)#计算S中的值
                for jj in range(S.shape[1]):#将S中的每一行进行归一化
                    S[ii, jj] = 1-S[ii, jj]/sum(S[ii,:])
            if f_function(F,S) < f_before:#目标函数值下降,进行下一次迭代
                S_before = S
                F = F_before
                f_before = f_function(F,S)
            else:#终止算法
                S = S_before
                F = F_before
                break
        else:#终止算法
            S = S_before
            F = F_before
            break
    model = PCA(n_components = k)#训练一个PCA模型，其降维的组成成分数目为k
    S = model.fit_transform(S)#用PCA模型对数据进行降维
    model = KMeans(n_clusters = k)#计算最终的聚类结果
    model.fit(S)
    result = np.array([model.labels_])
    return result

if __name__ == '__main__':
    name = 'glass'#数据集的名称
    dataFile = 'C:/Users/Administrator/Desktop/TR/datasets/' + name + '.mat'#数据集的路径
    data = sio.loadmat(dataFile)#读取数据集
    data = data[name]
    data = np.array(data)
    label = np.array([data[:,data.shape[1]-1]])
    data = data[:,0:data.shape[1]-1]
    #model = MinMaxScaler()
    #data = model.fit_transform(data)
    k =3#聚类的类簇数
    result = SEANC(data,k)
    R, FM, K, RT, precision, recall, F1, NMI, P= Evaluation.evaluation(data,result,label)
    print('R=',R)
    print('FM=',FM)
    print('K=', K)
    print('RT=', RT)
    print('precision=', precision)
    print('recall=', recall)
    print('F1=', F1)
    print('NMI=', NMI)
    print('P=', P)