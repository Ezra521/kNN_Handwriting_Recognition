"""
Handwriting recognition by kNN
Created on Dec 13, 2017
kNN: k Nearest Neighbors
@author: Ezra
@email:zgahwuqiankun@qq.com
其实就是判断预测的数据和训练的数据哪几个欧式距离最近，就判定手写体是那个数字
"""
import  numpy as np
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #计算dataSet里总共有多少样本
    #tile()是扩展函数，就是按照x轴或者y轴把数据复制
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet #利用tile函数扩展预测数据，与每一个训练样本求距离

    sqDiffMat = diffMat**2 #对求出来的差分别平方
    sqDistances = sqDiffMat.sum(axis=1)#sum应该是默认的axis=0 就是普通的相加，而当加入axis=1以后就是将一个矩阵的每一行向量相加
    distances = sqDistances**0.5  #求欧式距离
    sortedDistIndicies = distances.argsort()#argsort函数返回的是数组值从小到大的索引值,记住返回值是数组的下标，可以这么理解
    classCount={}  #定义一个字典，用于储存K个最近点对应的分类以及出现的频次
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1#字典里面的该标签数量++
    # 以下代码将不同labels的出现频次由大到小排列，输出次数最多的类别
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


#将一个32*32的二进制的图像矩阵转换成为1*1024的向量
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('./dataSet/trainingDigits')           #载入训练数据
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #去掉后缀.txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('./dataSet/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('./dataSet/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #去掉.txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./dataSet/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类器分类结果是: %d, 正确的结果是: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n总共判断错误的数据是: %d" % errorCount)
    print("\n总共的错误率是: %f" % (errorCount/float(mTest)))

handwritingClassTest()