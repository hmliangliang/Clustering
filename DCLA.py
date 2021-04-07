# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\7\3 0003 10:01:01
# File:         DCLA.py
# Software:     PyCharm
#------------------------------------
import numpy as np
import scipy.io as sio
import time
import math
import Evaluation
from sklearn.decomposition import PCA


def f_obj(W,X,S,F,G,lamda,gamma):#计算目标函数值
    value = 0
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            value = value + S[i,j]*np.linalg.norm(np.dot(W.transpose(),X[:,i]) - np.dot(W.transpose(),X[:,j]),2)**2
    value = value + gamma*np.linalg.norm(S,ord = 'fro')**2
    value = value + lamda*np.linalg.norm(np.dot(X.transpose(),W) - np.dot(F,G.transpose()),ord = 'fro')**2
    return value

def update_F(F,W,X,G):#更新F
    F = np.zeros((X.shape[1],G.shape[1]))#初始化F
    for i in range(X.shape[1]):#每一个样本
        num = 0  # 记录序号
        value = np.inf  # 记录最小值
        for j in range(k):#每一个类簇
            if np.linalg.norm(np.dot(W.transpose(),X[:,i])-G[:,j],2)**2 < value:#获得更小的距离
                value = np.linalg.norm(np.dot(W.transpose(),X[:,i])-G[:,j],2)**2
                num = j#获得更小的距离对应的类簇序号
        F[i,num] = 1
    return F


def DCLA(data,k,d):
    '''此算法执行的是Wang X D, Chen R C, Zeng Z Q, et al. Robust Dimension Reduction for Clustering With Local Adaptive
    Learning[J]. IEEE transactions on neural networks and learning systems, 2019,30(3): 657-668.
    data: 数据集每一行代表一个样本
    k:聚类的类簇数目
    d:降维后的维数
    return:返回每个样本的类标签1*n,类型为np.arrary([[]])
    '''
    gamma=1
    lamda=1#gamma,lamda:相关的参数
    data = np.array(data)
    X = data.transpose()#m*n大小
    #初始化相关的参数
    e = 0.000001#迭代的最大误差
    S = np.zeros((data.shape[0],data.shape[0]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if i!=j:
                S[i,j] = math.exp(-np.linalg.norm(X[:,i]-X[:,j],2))
    for i in range(X.shape[1]):
        value = sum(S[i,:])
        for j in range(X.shape[1]):
            S[i,j] = S[i,j]/value
    S_before = S
    F = np.zeros((data.shape[0],k))#n*k大小
    for i in range(data.shape[0]):#随机分配类标签
        F[i,np.random.randint(0,k,(1,1))[0,0]] = 1
    W = np.random.rand(data.shape[1],d)#m*d大小
    F_before = F
    W_before = W
    G = np.dot(W.transpose(),np.dot(X,np.dot(F,np.linalg.inv(np.dot(F.transpose(),F)))))
    G_before = G#大小d*c
    N_MAX = 50
    f = -np.inf
    f_before = np.inf
    for num in range(N_MAX):
        if f<= f_before:
            #更新F
            F = update_F(F,W,X,G)
            G = np.dot(W.transpose(),np.dot(X,np.dot(F,np.linalg.inv(np.dot(F.transpose(),F)))))
            #更新W
            Ds = np.zeros((X.shape[1],X.shape[1]))
            for i in range(X.shape[1]):#计算Eq.(14)的对角矩阵
                for j in range(X.shape[1]):
                    Ds[i,i] = Ds[i,i] + (S[i,j]+S[j,i])/2
            Ls = Ds - (S.transpose() + S)/2#计算新的拉普拉斯矩阵
            M = np.dot(X,np.dot(Ls,X.transpose())) + lamda*np.dot(X,X.transpose()) - lamda*np.dot(X,np.dot(F,np.dot(np.linalg.inv(np.dot(F.transpose(),F)),np.dot(F.transpose(),X.transpose()))))
            #对M进行特征值分解
            eigvalue,eigvector = np.linalg.eig(M)
            indx = np.argsort(eigvalue)#对特征值由大到小排序
            indx = indx[1:(d+1)]
            W = eigvector[:,indx]#更新W
            #更新S
            S = np.zeros((X.shape[1],X.shape[1]))#初始化S
            for i in range(X.shape[1]):
                for j in range(X.shape[1]):
                    label = np.argwhere(F[i,:]==1)[0,0]#获取第i个样本的类标签
                    if F[i,label] == F[j,label]:#两个对象处于同一类标签
                        S[i,j] = 1/(len(np.argwhere(F[:,label]==1)))
                    else:#两个对象处于不同的类标签
                        S[i, j] = 0
            for i in range(X.shape[1]):#对S进行归一化
                value = sum(S[i, :])
                for j in range(X.shape[1]):
                    S[i, j] = S[i, j] / value
            f = f_obj(W,X,S,F,G,lamda,gamma)#计算新的目标函数值
            if abs(f - f_before) <= e:#目标函数值下降,更新参数进行下一次迭代
                f_before = f
                S_before = S
                W_before = W
                F_before = F
                G_before = G
            else:#函数值不下降,终止迭代过程
                F = F_before
                W = W_before
                G = G_before
                break
        else:#函数值不下降,终止迭代过程
            F = F_before
            W = W_before
            G = G_before
            break
    result = np.zeros((1,data.shape[0]))#初始化result
    for i in range(F.shape[0]):#获得最终的类标签
        result[0,i] = np.argwhere(F[i,:] == 1)[0,0]
    return result

if __name__ == '__main__':
    start = time.time()
    name = 'glass'  # 数据集的名称
    dataFile = 'C:/Users/Administrator/Desktop/TR/datasets/' + name + '.mat'  # 数据集的路径
    data = sio.loadmat(dataFile)  # 读取数据集
    data = data[name]
    data = np.array(data)
    label = np.array([data[:, data.shape[1] - 1]])
    data = data[:, 0:data.shape[1] - 1]
    model = PCA(n_components=int(0.8 * data.shape[1]))
    data = model.fit_transform(data)
    k = 2  # 聚类的类簇数
    d = int(0.8 * data.shape[1])
    result = DCLA(data,k,d)
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