# -*- coding: utf-8 -*-
"""
PR Project 3
LDA, LDA+PCA
author: Tmn07
date: 2016-12-25
"""
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from PIL import Image
from sklearn.decomposition import PCA
# classify using kNN  
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row

    ## step 1: calculate Euclidean distance  
    # tile(A, reps): Construct an array by repeating A reps times  
    # the following copy numSamples rows for dataSet  
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    ## step 2: sort the distance  
    # argsort() returns the indices that would sort an array in a ascending order  
    sortedDistIndices = np.argsort(distance)

    classCount = {}  # define a dictionary (can be append element)
    for i in xrange(k):
        ## step 3: choose the min k distance  
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur  
        # when the key voteLabel is not in dictionary classCount, get()  
        # will return 0  
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        ## step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex

def read_img(snum,pnum):
	"""读取一张图片上的数据
	Paramters:
		snum : 第几个人的
		pnum : 第几张照片
	Return:
		ndarray, shape=(1,10304)
	"""
	with Image.open("att_faces/s"+str(snum)+"/"+str(pnum)+".pgm") as im:
		# print im.size
		a = np.array(im)
		return a.reshape(1,10304)

def read_data(sid,eid):
	"""批量读取照片
	Paramters:
		sid : 从某人的第几张照片开始
		pnum : 到某人的第几张照片介绍
	Return:
		ndarray, shape=(40*(eid-sid+1),10304)
	"""
	X = read_img(1,sid)
	for i in range(1,41):
		for j in range(sid,eid+1):
			if i==1 and j ==sid:
				continue
			X = np.concatenate((X,read_img(i,j)))
	return X

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

def main():
	
	X = read_data(1,5)
	y = np.array([i/5+1 for i in range(0,200)])
	lda = LinearDiscriminantAnalysis() # 初始化一个LDA模型，默认参数
	train_res = lda.fit(X, y) # 训练
	X_r1 = train_res.transform(X) # 对X降维

	print X_r1.shape,y.shape

	X2 = read_data(6,10)
	X_r2 = train_res.transform(X2)
	plabels = []
	for Xone in X_r2:
		outputLabel = kNNClassify(Xone, X_r1, y, k=5)
		plabels.append(outputLabel)
	print precent(y,plabels)
	# print train_res.score(X2,y)


	pca_list = [10,20,30,40,50,100,200,300,1000,2000,3000,10000]

	for nc in [43]:
	# for nc in pca_list:
	# for nc in range(30,100):
		pca = PCA(n_components=nc)
		trained_pca = pca.fit(X)
		# print(sum(pca.explained_variance_ratio_) )
		X_r1 = trained_pca.transform(X)

		lda = LinearDiscriminantAnalysis()
		train_res = lda.fit(X_r1, y)
		X_r1 = train_res.transform(X_r1)

		X_r2 = trained_pca.transform(X2)
		X_r2 = train_res.transform(X_r2)

		plabels = []
		for Xone in X_r2:
			outputLabel = kNNClassify(Xone, X_r1, y, k=5)
			plabels.append(outputLabel)
		print "PCA n_components =",nc
		print precent(y,plabels)

if __name__ == '__main__':

	main()