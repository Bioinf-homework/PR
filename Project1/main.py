# coding=utf-8
from numpy import *
from sklearn.decomposition import PCA

def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(",")
        lineArr.append(1.0) # 加入偏置
        dataMat.append(lineArr[1:])

        labelMat.append(lineArr[0])
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def precent(y,pre_y):
    """判断预测准确率
    paramters:
        y : n*1 ndarray or list
        pre_y : n*1 ndarray or list
    Return:
        float, the right predict precent
    """
    n = len(y)
    r = 0
    for i in xrange(n):
        if y[i] == pre_y[i]:
            r += 1
    return 1.0*r/n

def stocGradAscent(dataMatrix, classLabels):
    m,n = shape(dataMatrix) # 获取矩阵大小
    alpha = 1 # 步长
    lam = 80 # 正则化项系数
    weights = ones(n) # 初始化参数
    for j in range(40):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            # 随机一条数据
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = h - classLabels[randIndex] 
            weights = weights - alpha * error * dataMatrix[randIndex]
            # weights = weights - alpha * error * dataMatrix[randIndex] - alpha*lam/m*weights
            del(dataIndex[randIndex])
    return weights


data,label = loadDataSet('SPECTF.train.csv')
dataArr = array(data, dtype=float64)
labelArr = array(label, dtype=float64)

w = stocGradAscent(dataArr,labelArr)

# 读取测试集
tdata,tlabel = loadDataSet('SPECTF.test.csv')
tdataArr = array(tdata, dtype=float64)
tlabelArr = array(tlabel, dtype=float64)

# 计算结果
res = sigmoid(tdataArr.dot(w))

# 分类
for i in range(len(res)):
    if res[i] > 0.5:
        res[i] = 1
    else:
        res[i] = 0

# 输出结果
# print res
print precent(tlabelArr,res)