
from math import log
import operator
import matplotlib.pyplot as plt
from pandas import DataFrame as df
import pandas as pd

def createDataSet():
    data = pd.read_csv("water_melon_2_0.data")
    dataSet = data.get(["色泽","根蒂","敲声","纹理","脐部","触感","好瓜"])
    dataSet = dataSet.values.tolist()
    labels = ["色泽","根蒂","敲声","纹理","脐部","触感"]
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 样本数
    labelCounts = {}   # 创建一个数据字典：key是最后一列的数值（即标签，也就是目标分类的类别），value是属于该类别的样本个数
    for featVec in dataSet: # 遍历整个数据集，每次取一行
        currentLabel = featVec[-1]  #取该行最后一列的值
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0  # 初始化信息熵
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2  计算信息熵
    return shannonEnt

def splitDataSet(dataSet, axis, value): 
    # axis是dataSet数据集下要进行特征划分的列号
    # value是该列下某个特征值
    # dataSet 传入的数据集
    retDataSet = []

    #遍历数据集，并抽取按axis的当前value特征进划分的数据集(不包括axis列的值)
    for featVec in dataSet: 
        if featVec[axis] == value: #
            reducedFeatVec = featVec[:axis]     
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#计算每种划分方式的信息熵
def calculate_max_entropy(featList,dataSet,feature_num):
    uniqueVals = set(featList)
    newEntropy = 0.0
    for value in uniqueVals: 
        subDataSet = splitDataSet(dataSet, feature_num, value)
        prob = len(subDataSet)/float(len(dataSet))
        newEntropy += prob * calcShannonEnt(subDataSet)
    return newEntropy

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #获取当前数据集的特征个数，最后一列是分类标签
    baseEntropy = calcShannonEnt(dataSet)  #计算当前数据集的信息熵
    bestInfoGain = 0.0; bestFeature = -1   #初始化最优信息增益和最优的特征
    for i in range(numFeatures):        #遍历每个特征iterate over all the features
        featList = [example[i] for example in dataSet]#获取数据集中当前特征下的所有值
        newentropy = calculate_max_entropy(featList,dataSet,i)
        infoGain = baseEntropy - newentropy     #计算信息增益
        if (infoGain > bestInfoGain):       #比较每个特征的信息增益，只要最好的信息增益
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature         


## 无法区分时，满足书中条件2，判断叶子节点类别             
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def countAttributeTypes(dataSet):
    dataSet=df(dataSet)
    return set(dataSet.iloc[:,:-1])
    

def treeGenerate(dataSet,labels,original_dataSet,original_labels):
    classList = [example[-1] for example in dataSet] # 返回当前数据集下标签列所有值
    if classList.count(classList[0]) == len(classList):
        return classList[0]#当类别完全相同时则停止继续划分，直接返回该类的标签
    if len(dataSet[0]) == 1 or countAttributeTypes(dataSet=dataSet) == 1: ##遍历完所有的特征时，仍然不能将数据集划分成仅包含唯一类别的分组 dataSet
        return majorityCnt(classList) #由于无法简单的返回唯一的类标签，这里就返回出现次数最多的类别作为返回值
    bestFeat = chooseBestFeatureToSplit(dataSet) # 获取最好的分类特征索引
    bestFeatLabel = labels[bestFeat] 
    bestFeat = original_labels.index(bestFeatLabel)
    myTree = {bestFeatLabel:{}} #当前数据集选取最好的特征存储在bestFeat中
    featValues = [example[bestFeat] for example in original_dataSet]
    uniqueVals = set(featValues)
    bestFeat = labels.index(bestFeatLabel)
    for value in uniqueVals:
        if splitDataSet(dataSet, bestFeat, value) == []:
            myTree[bestFeatLabel][value] =  majorityCnt(classList)
        else:
            subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
            del(subLabels[bestFeat])
            myTree[bestFeatLabel][value] = treeGenerate(splitDataSet(dataSet, bestFeat, value),subLabels,original_dataSet,original_labels)
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel



decisionNode = dict(boxstyle="sawtooth", fc="0.8") #定义文本框与箭头的格式
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(myTree): #获取树叶节点的数目
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#测试节点的数据类型是不是字典，如果是则就需要递归的调用getNumLeafs()函数
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree): #获取树的深度
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #计算树的宽度
    depth = getTreeDepth(myTree)   #计算树的深度
    firstStr = list(myTree.keys())[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()




if __name__ == '__main__':
    #!/usr/bin/env python
    #coding:utf-8
    """a demo of matplotlib"""
    import matplotlib as  mpl
    from matplotlib  import pyplot as plt
    mpl.rcParams[u'font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False
    import matplotlib
    matplotlib.rcParams['axes.unicode_minus']=False
    lenses,lensesLabels = createDataSet()
    lensesTree = treeGenerate(lenses,lensesLabels,lenses,lensesLabels)
    createPlot(lensesTree)
