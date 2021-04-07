# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\6\5 0005 09:45:45
# File:         ESEC.py
# Software:     PyCharm
#------------------------------------

import numpy as np
import math
import scipy.io as sio
import Evaluation
from sklearn.cluster import KMeans
import time
from sklearn.preprocessing import MinMaxScaler

def Laplician(data,k):#计算拉普拉斯矩阵
    k=2*k
    W = np.zeros((data.shape[0],data.shape[0]))
    D = np.zeros((data.shape[0],data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            W[i,j] = math.exp(-np.linalg.norm(data[i,:]-data[j,:],2)/(2*data.shape[1]))#计算W中的元素值
        W_temp = W[i,:]
        W_temp = -np.sort(W_temp)#对距离值进行排序
        for j in range(min(2*k,len(W_temp)),len(W_temp)):
            W[i,np.argwhere(W[i,:]==W_temp[j])] = 0
        D[i,i] = sum(W[i,:])
    L = D - W#计算拉普拉斯矩阵
    return L,D

def ESEC(data,k,n_neuro,fun):
    '''
    此算法执行的是Liu M, Liu B, Zhang C, et al. Spectral nonlinearly embedded clustering algorithm[J]. Mathematical Problems in Engineering, 2016, 9264561:1-10.
    data:输入的数据是一个n*m的矩阵,每一行代表一个样本
    k:聚类的类簇数
    L:隐含层节点的数目
    fun:隐含层的节点的激活函数
    return:聚类的结果result,每一列代表一个样本,形式为np.arrary([[]])
    '''
    lamda = 1
    u = 0.01
    data = np.array(data)
    #随机生成ELM的参数
    a = np.random.rand(data.shape[1],n_neuro)
    b = np.random.rand(1,n_neuro)
    temph = np.dot(data,a) + b
    #计算隐含层节点的输出值
    if fun == 'sigmoid':
        H = 1/(1 + np.exp(-temph))
    elif fun == 'sin':
        H = np.sin(temph)
    elif fun == 'relu':
        for i in range(temph.shape[0]):
            for j in range(temph.shape[1]):
                H[i,j] = max(0,temph[i,j])
    elif fun == 'tanh':
        H = (np.exp(temph)-np.exp(-temph))/(np.exp(temph) + np.exp(-temph))
    #计算矩阵Hn
    if n_neuro <= data.shape[0]:#隐含层节点数目小于样本数
        Lh = np.eye(data.shape[0]) - np.dot(H,np.dot(np.linalg.pinv(lamda*np.eye(n_neuro) + np.dot(H.transpose(),H)),H.transpose()))
    else:#隐含层节点数目大于样本数
        Lh = np.eye(data.shape[0]) - np.dot(H,np.dot(H.transpose(),np.linalg.inv(lamda*np.eye(data.shape[0]) + np.dot(H,H.transpose()))))
    L,D = Laplician(data,k)#计算拉普拉斯矩阵
    DD = np.zeros((data.shape[0],data.shape[0]))#初始化矩阵
    for i in range(data.shape[0]):
        DD[i,i] = 1/math.sqrt(D[i,i])
    Lbar = np.dot(DD,np.dot(L,DD))
    eigvalues,eigvectors = np.linalg.eig(Lbar + u*Lh)#计算特征值和特征向量
    seq = np.argsort(eigvalues)
    seq = seq[1:k+1]
    F = eigvectors[:,seq]#获得最终的特征向量
    model = KMeans(n_clusters=k)#利用k-means算法来对数据进行聚类
    model.fit(F)
    result = np.array([model.labels_])#获取聚类的类标签
    return result

if __name__ == '__main__':
    start = time.time()
    name = 'glass'#数据集的名称
    dataFile = 'C:/Users/Administrator/Desktop/TR/datasets/' + name + '.mat'#数据集的路径
    data = sio.loadmat(dataFile)#读取数据集
    data = data[name]
    data = np.array(data)
    label = np.array([data[:,data.shape[1]-1]])
    data = data[:,0:data.shape[1]-1]
    #model = MinMaxScaler()
    #data = model.fit_transform(data)
    k = 6#聚类的类簇数
    n_neuro = int(data.shape[0]/2)
    fun = 'sigmoid'
    result = ESEC(data,k,n_neuro,fun)
    R, FM, K, RT, precision, recall, F1, NMI,P= Evaluation.evaluation(data,result,label)
    print('R=',R)
    print('FM=',FM)
    print('K=', K)
    print('RT=', RT)
    print('precision=', precision)
    print('recall=', recall)
    print('F1=', F1)
    print('NMI=', NMI)
    print('P=', P)
    end = time.time()
    print('算法的时间消耗为:',end-start)