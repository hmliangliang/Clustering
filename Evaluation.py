import numpy as np
import math
from scipy.special import comb
from sklearn import metrics

def evaluation(data,result,label):#注输入向量均为numpy.array类型
    '''此函数主要是计算各种评价指标
    data:图的邻接矩阵 data[i,j]=1表示第i个节点与第j个节点之间存在边的连接 data[i,j]=0表示第i个节点与第j个节点之间不存在边的连接
    result:表示聚类的结果,其是一个行向量1*n,每一列代表一个样本的类标签,np.array([[]])
    label:表示数据的真实类标签,其是一个行向量1*n,每一列代表一个样本的类标签
    '''
    if len(result) > 0:
        while min(result[0,:]) > 0:#把聚类的结果矩阵转变为0开始
            result[0, :] = result[0,:] -1
    if len(label) > 0:
        while min(label[0, :]) > 0:  # 把聚类的结果矩阵转变为0开始
            label[0, :] = label[0, :] -1
    #主要计算各种度量指标
    if len(label)!=0:#label为不为空
        TP=0#初始化
        FP=0
        FN=0
        TN=0
        for i in range(result.shape[1]-1):
            for j in range(i+1,result.shape[1]):
                if result[0,i]==result[0,j]:#两个样本聚在同一个类簇中
                    if label[0,i]==label[0,j]:#两个样本在实际也处于同一类簇中
                        TP=TP+1
                    else:#两个样本在实际不处于同一类簇中
                        FP=FP+1
                else:#两个样本聚不在同一个类簇中
                    if label[0,i]==label[0,j]:#两个样本在实际处于同一类簇中
                        FN=FN+1
                    else:#两个样本在实际不处于同一类簇中
                        TN=TN+1
        #TP = comb(TP,2)
        #FP = comb(FP, 2)
        #FN = comb(FN, 2)
        #TN = comb(TN, 2)
        N = FP + TP + FP + FN
        R = (TP+TN)/(TP+FP+FN+TN)
        FM=TP/math.sqrt((TP+FN)*(TP+FP))#计算的是Folkes and Mallows指标[0,1],值越大则结果越好
        #ARI=2*(TP*TN-FN*FP)/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN))#计算的是ARI指标[-1,1],值越大越好,-1表示完全不一致
        #Phi=(TP*TN-FN*FP)/math.sqrt((TP+FN)*(TP+FP)*(FN+TN)*(FP+TN))#计算的是Phi指标[-1,1],值越大越好,-1表示完全不一致
        #Hubert=(N*TP-(TP+FN)*(TP+FP))/math.sqrt((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP))#计算的是Hubert指标[-1,1],值越大越好,-1表示完全不一致
        K=0.5*(TP/(TP+FP)+TP/(TP+FN))#计算的是Kulczynski指标[0,1],值越大越好,0表示完全不一致,1表示完全一致
        RT=(TP+TN)/(TP+TN+2*(FN+FP))#计算的是Rogers-Tanimoto指标[0,1]
        precision=TP/(TP+FP)#计算的是precision指标
        recall=TP/(TP+FN)#计算的是召回率指标
        F1=(2*precision*recall)/(precision+recall)#计算的是F-measure指标
        #计算互信息度量指标
        NMI=0.0#互信息值
        cresult=[]#将result改写成列表的形式，列表中的每一个元素即为一个类簇样本的序号
        clabel=[]#将label改写成列表的形式，列表中的每一个元素即为一个类簇样本的序号
        #聚类结果找出不同类标签
        values1=np.unique(result[0,:])
        values1=values1.reshape((1,values1.shape[0]))
        values2=np.unique(label[0,:])
        values2 = values2.reshape((1, values2.shape[0]))
        for i in range(values1.shape[1]):
            temp=[]
            temp=np.argwhere(result[0,:]==values1[0,i]).reshape(1,np.argwhere(result[0,:]==values1[0,i]).shape[0])#找出聚类结果中每一类簇的样本保存到列表中
            cresult.append(temp[0,:].tolist())
        for i in range(values2.shape[1]):
            temp=[]
            temp=np.argwhere(label[0,:]==values2[0,i]).reshape(1,np.argwhere(label[0,:]==values2[0,i]).shape[0])#找出真实类标签中每一类簇的样本保存到列表中
            clabel.append(temp[0,:].tolist())
        del values1
        del values2
        value1=0#公式上面的求和值
        value2=0#公式下面的第一个求和值
        value3=0#公式下面的第二个求和值
        for iset in cresult:
            Ni=len(iset)#聚类的类簇Ci的大小
            if Ni>0:
                value2=value2+Ni*math.log(Ni/data.shape[0])
            for jset in clabel:
                Nj=len(jset)#真实类簇的大小
                Nij=np.array([np.intersect1d(jset,iset)]).shape[1]#两个集合的交集
                if Ni>0 and Nj>0:
                    value1=value1+Nij*math.log((data.shape[0]*Ni)/(Ni*Nj))
                    value3=value3+Nj*math.log(Nj/data.shape[0])
        if value2==0 or value3==0:
            NMI = 0
        else:
            NMI = value1/math.sqrt(value2*value3)  # 计算最终的互信息值
        NMI = 1/(1+np.exp(-NMI))
        #计算朴素度
        P = 0
        CLABEL1 = np.unique(result[0])
        CLABEL2 = np.unique(label[0])
        for i in range(len(CLABEL1)):
            s=[]
            for j in range(len(CLABEL2)):
                s.append(len(np.intersect1d(np.argwhere(result[0] == CLABEL1[i]),np.argwhere(label[0] == CLABEL2[j]))))
            P = P + max(s)
        P = P/len(result[0])
        '''#计算标准互信息,计算方法见https://blog.csdn.net/DreamHome_S/article/details/78379635
        HX = 0
        HY = 0
        HXY = 0
        value1 = np.unique(result[0,:])
        value2 = np.unique(label[0,:])
        for i in range(len(value1)):
            HX = HX - len(np.argwhere(result[0,:]==value1[i]))/len(result[0,:])*math.log(len(np.argwhere(result[0,:]==value1[i]))/len(result[0,:]))
        for j in range(len(value2)):
            HY = HY - len(np.argwhere(label[0, :] == value2[j]))/len(label[0, :]) * math.log(len(np.argwhere(label[0, :] == value2[j]))/len(label[0, :]))
        for i in range(len(value1)):
            Pi = len(np.argwhere(result[0,:]==value1[i]))/len(result[0,:])
            for j in range(len(value2)):
                Pj = len(np.argwhere(label[0, :] == value2[j]))/len(label[0, :])
                Pij = len(np.intersect1d(np.argwhere(label[0, :] == value2[j]),np.argwhere(result[0,:]==value1[i])))/len(result[0,:])
                HXY = HXY + Pij*math.log(Pij/(Pi*Pj))
        NMI = 2*HXY/(HX+HY)'''
        return R,FM,K,RT,precision,recall,F1,NMI,P#返回计算的结果
    else:
        return 0,0,0,0,0,0,0,0,0,0
