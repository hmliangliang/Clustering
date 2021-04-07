# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\6\7 0007 15:50:50
# File:         FSCM.py
# Software:     PyCharm
#------------------------------------
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import Evaluation
import scipy.io as sio
import time

def FSCM(data,k):
    '''
    执行的是Bian Z, Ishibuchi H, Wang S. Joint Learning of Spectral Clustering Structure and Fuzzy Similarity Matrix of Data[J].
    IEEE Transactions on Fuzzy Systems, 2019, 27(1): 31-44.
    data:输入的数据,每一行代表一个样本
    k:聚类的类簇数目
    return:返回聚类的结果,每一列代表一个样本的类簇标签
    '''
    #配置相应的参数
    data = np.array(data)
    beta = 30
    e = math.pow(10,-12)
    r = 0.6
    N_MAX = 100#最大迭代次数
    #初始化temp1与temp2
    temp1 = np.inf
    temp2 = -np.inf
    S = np.zeros((data.shape[0],data.shape[0]))#初始化相似度矩阵
    for i in range(data.shape[0]):#计算相似度矩阵,按照Eq.(14)式计算
        value = 0
        for j in range(data.shape[0]):
            if i!=j:#保证对角矩阵的元素为0
                S[i,j] = math.exp(-np.linalg.norm(data[i,:]-data[j,:],2))
                value = value + S[i,j]
        for j in range(data.shape[0]):
            S[i,j] = math.pow(S[i,j]/value,1/r)
    num = 0
    while temp1> e or temp2 < e:
        num = num +1
        if num > N_MAX:#迭代次数达到上限
            break
        else:
            Ds = np.zeros((data.shape[0],data.shape[0]))#初始化对角矩阵
            for i in range(data.shape[0]):#计算对角矩阵
                for j in range(data.shape[0]):
                    Ds[i,i] = Ds[i,i] + (S[i,j] +S[j,i])/2
            Ls = Ds - (S + S.transpose())/2#求出拉普拉斯矩阵
            eigvalues,eigvectors = np.linalg.eig(Ls)#对Ls进行特征值分解
            seq = np.argsort(eigvalues)#对特征值进行排序
            seq = seq[1:k+1]#前k个最小的非零特征值的序号
            temp1 = sum(eigvalues[seq])#前k个最小的非零特征值的和
            F = eigvectors[:,seq]#前k个最小的非零特征值对应的特征向量
            #按照Eq.(30)更新相似度矩阵S
            for i in range(S.shape[0]):
                for j in range(S.shape[1]):
                    if i!=j:
                        value = 0
                        for kk in range(S.shape[0]):
                            if kk != i:
                                value = value + math.pow((math.pow(np.linalg.norm(data[i,:]-data[kk,:],2),2)+beta*(math.pow(np.linalg.norm(F[i,:]-F[kk,:],2),2)))/(math.pow(np.linalg.norm(data[i,:]-data[j,:],2),2)+beta*(math.pow(np.linalg.norm(F[i,:]-F[j,:],2),2))),r/(r-1))
                        S[i,j] = 1/math.pow(value,1/r)
            if temp1 > e:#更新参数beta
                beta = 2*beta
            #再次计算Ls
            Ds = np.zeros((data.shape[0], data.shape[0]))  # 初始化对角矩阵
            for i in range(data.shape[0]):  # 计算对角矩阵
                for j in range(data.shape[0]):
                    Ds[i, i] = Ds[i, i] + (S[i, j] + S[j, i]) / 2
            Ls = Ds - (S + S.transpose()) / 2  # 求出拉普拉斯矩阵
            eigvalues, eigvectors = np.linalg.eig(Ls)  # 对Ls进行特征值分解
            seq = np.argsort(eigvalues)  # 对特征值进行排序
            seq = seq[1:k + 1]  # 前k个最小的非零特征值的序号
            temp2 = sum(eigvalues[seq])  # 前k个最小的非零特征值的和
            if temp2 < e:#更新参数beta
                beta = beta/2
    result = np.zeros((1,data.shape[0]))#初始化聚类结果矩阵
    for i in range(F.shape[0]):#获取数据的聚类结果
        result[0,i] = np.argwhere(F[i,:] == max(F[i,:]))[0,0]
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
    model = PCA(n_components=int(0.8 * data.shape[1]))
    data = model.fit_transform(data)
    model = MinMaxScaler()
    data = model.fit_transform(data)
    k = 3  # 聚类的类簇数
    result = FSCM(data, k)
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
