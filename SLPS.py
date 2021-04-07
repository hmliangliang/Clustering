# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2019\7\2 0002 09:44:44
# File:         SLPS.py
# Software:     PyCharm
#------------------------------------
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import scipy.io as sio
import time
import math
import Evaluation
from sklearn.decomposition import PCA

def x_to_y(data):
    '''将X中每一个元素拼接成一行'''
    value=np.zeros((1,data.shape[0]*data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value[0,i*data.shape[1]+j] = data[i,j]
    return value

def y_to_x(data,n,m):
    '''将拼接的元素还原成原来的形式'''
    value = np.zeros((n,m))
    for i in range(data.shape[1]):
        value[int(i/m),i%m] = data[0,i]
    return value

def f_obj(X,Y,W,P,alpha,beta):#计算目标函数
    '''X:n*m型; Y:n*m型; W:n*n型; P:m*d型'''
    value1 = 0  #Eq.(12)中第一项的和
    value2 = 0  #Eq.(12)中第二项的和
    value3 = 0  #Eq.(12)中第三项的和
    for i in range(Y.shape[0]):
        value2 = value2 + alpha * np.linalg.norm(X[i, :] - Y[i, :], 2) ** 2
        for j in range(Y.shape[1]):
            value1 = value1 + (1-alpha)*W[i,j]*np.linalg.norm(Y[i,:]-Y[j,:],2)**2
    Z = np.dot(Y,P)#计算变换后的样本
    Z = np.mean(Y,axis=0)
    for i in range(Y.shape[0]):
        value3 = value3 + beta*np.linalg.norm(Y[i,:]-Z,2)**2
    return value1+value2-value3

def SLPS(data,d, k, delta, alpha,beta):
    '''
    本算法执行的是Hou C, Nie F, Jiao Y, et al. Learning a subspace for clustering via pattern shrinking[J]. Information
    Processing & Management, 2013, 49(4): 871-883.
    data:输入的样本,每一行代表一个样本,每一列代表一个特征
    d:降维后的数据的维度
    k:聚类的类簇数目
    delta:原文Eq.(11)中的参数
    alpha:原文Eq.(11)中的参数
    beta:原文Eq.(11)中的参数
    result:返回聚类的类簇,其中result是一个np.arrary([[]])类型n*1,每一个元素表示当前样本的类簇标签
    '''
    data = np.array(data)
    result = np.zeros((1,data.shape[0]))#初始化聚类结果
    #计算样本的权值
    W = np.zeros((data.shape[0],data.shape[0]))#初始化权值
    for i in range(W.shape[0]):#计算权值
        for j in range(W.shape[1]):
            W[i,j] = math.exp(-np.linalg.norm(data[i,:]-data[j,:],2)**2/(2*delta**2))
            W[j,i] = W[i,j]
    for i in range(W.shape[0]):#计算权值
        value = np.mean(W[i,:])
        for j in range(W.shape[1]):
            if W[i,j] <= 0.8*value:
                W[i,j] = 0
                W[j,i] = W[i,j]
    D = np.zeros((data.shape[0],data.shape[0]))#计算对角矩阵
    for i in range(D.shape[0]):
        D[i,i] = sum(W[:,i])
    L = D - W#计算拉普拉斯矩阵
    N_MAX = 50#算法最大迭代次数
    #初始化参数值
    P = np.random.rand(data.shape[1],d)#初始化矩阵P
    Y = np.random.rand(data.shape[1],data.shape[0])
    f = -np.inf
    f_before = np.inf
    P_before = P
    Y_before = Y
    H = np.eye(data.shape[0])-1/data.shape[0]*np.ones((data.shape[0],data.shape[0]))#计算中心化矩阵
    for num in range(N_MAX):
        if f <= f_obj(data,Y.transpose(),W,P,alpha,beta):
            #固定P,更新Y
            M = np.dot(P,P.transpose())
            x = x_to_y(data)#把原数据转换成一行形式
            y = np.dot(x,np.linalg.inv(np.kron((1-alpha)*L,np.eye(data.shape[1])) + alpha*np.eye(data.shape[0]*data.shape[1]) - beta*np.kron(H,M)))#按照Eq.(16)式更新y
            Y = y_to_x(y,data.shape[0],data.shape[1])#把y数据转换成多行多列的形式
            Y = Y.transpose()
            #固定Y,更新P
            eigenvalue,eigenvector = np.linalg.eig(np.dot(np.dot(Y,H),Y.transpose()))#计算特征值与特征向量
            indx = np.argsort(-eigenvalue)
            indx = indx[0:d]#获取d个最大特征值所对应的特征向量
            P = eigenvector[:,indx]
            f = f_obj(data,Y.transpose(),W,P,alpha,beta)
            if f < f_before:#目标函数值下降,进行下一次迭代
                f_before = f
                P_before = P
                Y_before = Y
            else:#目标函数值不下降,终止迭代
                P = P_before
                Y = Y_before
                break
        else:#目标函数值不下降,终止迭代
            P = P_before
            Y = Y_before
            break
    Y = np.dot(P.transpose(),Y)#获取最终的降维数据
    Y = Y.transpose()
    model = KMeans(n_clusters=k)
    model.fit(Y)
    result = np.array([model.labels_])
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
    k = 2  # 聚类的类簇数
    d = int(0.8*data.shape[1])
    delta = 10
    alpha = 0.15
    beta = 0.6
    result = SLPS(data,d, k, delta, alpha,beta)
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
