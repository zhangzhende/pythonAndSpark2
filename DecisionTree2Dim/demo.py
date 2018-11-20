from pyspark import SparkContext
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.recommendation import ALS
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from time import time
import pandas as pd
import matplotlib.pyplot as plt

sc = SparkContext('local')
global Path
Path = "D:\\workingSpace\\pythonAndSpark2\data\\stumbleupon\\"


def getData():
    print("开始导入数据。。。")
    path = Path + "train.tsv"
    print(path)
    # 使用minPartitions=40，将数据分成40片，不然报错
    rawDataWithHeader = sc.textFile(path, minPartitions=40)
    header = rawDataWithHeader.first()
    # 去掉首行，标题
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    # 去掉引号
    rData = rawData.map(lambda x: x.replace("\"", ""))
    # 按照制表符分字段
    lines = rData.map(lambda x: x.split("\t"))
    print("总共有:", str(lines.count()))
    print("前两个：", rawDataWithHeader.take(2))
    # 网页分类字典,
    # map(lambda fields: fields[3])：读取第三个字段
    # distinct()：保留不重复的数据
    # zipWithIndex将第三个字段中不重复的数据进行编号
    # categoriesMap的数据类型为dict字典，{"?":12,"business":2...}这样
    categoriesMap = lines.map(lambda fields: fields[3]).distinct().zipWithIndex().collectAsMap()
    print("categoriesMap字典：", categoriesMap)
    print("categoriesMap的项数：", len(categoriesMap))
    print("categoriesMap的类型", type(categoriesMap))
    # 创建LabeldPoint格式格式的数据
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extractFeatures(r, categoriesMap, len(r) - 1)))
    print("labelpointRDD 第一条数据：", labelpointRDD.take(1))
    #数据集划分比例8:1:1
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8,1,1])
    print("数据集划分为：trainData:",str(trainData.count()),"validationData:",str(validationData.count()),"testData:",str(testData.count()))

    #将数据暂存内存中
    trainData.persist()
    validationData.persist()
    testData.persist()

    #训练模型
    """
    trainClassifier(cls, data, numClasses, categoricalFeaturesInfo, impurity="gini", maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0)
    data:输入的训练数据
    numClasses:分类数目
    categoricalFeaturesInfo：设置分类的特征字段信息，本例采用OneHotEncoding故而传{}
    impurity：评估方式entropy熵，gini基尼系数
    maxDepth：决策树最大深度
    maxBins：每个节点的最大分支数
    """
    model=DecisionTree.trainClassifier(data=trainData,numClasses=2,categoricalFeaturesInfo={},impurity="entropy",maxDepth=5,maxBins=5)
    #计算AUC（ROC曲线下的面积）
    score=model.predict(validationData.map(lambda x:x.features))
    print(score)
    scoreAndLabels=score.zip(validationData.map(lambda x:x.label))
    print("scoreAndLabels的前5项",scoreAndLabels.take(5))
    metrics=BinaryClassificationMetrics(scoreAndLabels)
    AUC=metrics.areaUnderROC
    print("AUC:",AUC)
    #评估impurity参数
    impurityList=["gini","entropy"]
    maxDepthList=[10]
    maxBinsList=[10]
    metrics2=[trainEvaluateModel(trainData,validationData,impurty,maxDepth,maxBins)
              for impurty in impurityList
              for maxDepth in maxDepthList
              for maxBins in maxBinsList]
    print("metrics2:",metrics2)
    #结果转换为图表，利用pandas
    indexList=impurityList
    df =pd.DataFrame(data=metrics2,index=indexList,columns=["AUC","duration","impurty","maxDepth","maxBins","model"])
    print("df:\n",df)
    showChart(df,"impurity","AUC","duration",0.5,0.7)
    #评估maxDepth
    evalParammeter(trainData,validationData,"maxDepth",impurtyList=["gini"],maxDepthList=[3,5,10,15,20],maxBinsList=[10])
    #评估maxbins
    evalParammeter(trainData, validationData, "maxBins", impurtyList=["gini"], maxDepthList=[10],
                   maxBinsList=[3,5,10,50,100,200])
    #找出最佳的参数组合
    bestModel=evalAllParammeter(trainData,validationData,
                                impurityList=["gini","entropy"],
                                maxDepthList=[3,5,10,15,20],
                                maxBinsList=[3,5,10,50,100])

    # return rawDataWithHeader
#评估所有参数，选择最佳参数组合
def evalAllParammeter(trainData,validationData,impurityList,maxDepthList,maxBinsList):
    #for循环遍历所有参数集合
    metrics = [trainEvaluateModel(trainData, validationData, impurty, maxDepth, maxBins)
                for impurty in impurityList
                for maxDepth in maxDepthList
                for maxBins in maxBinsList]
    #找出AUC最大的参数组合
    Smetrics=sorted(metrics,key=lambda k:k[0],reverse=True)
    bestParammeter=Smetrics[0]
    #显示调校后的最佳参数组合
    print("调教的最佳参数: impurity=",bestParammeter[2],",maxDepth=",bestParammeter[3],",maxBins=",bestParammeter[4],"结果AUC=",bestParammeter[0])
    #返回最佳模型
    return bestParammeter[5]

#评估参数通用方法
def evalParammeter(trainData,validationData,evalParam,impurityList,maxDepthList,maxBinsList):
    #训练评估参数
    metrics = [trainEvaluateModel(trainData, validationData, impurty, maxDepth, maxBins)
                for impurty in impurityList
                for maxDepth in maxDepthList
                for maxBins in maxBinsList]
    #设置当前评估的参数
    if evalParam=="impurity":
        indexList=impurityList
    elif evalParam=="maxDepth":
        indexList=maxDepthList
    elif evalParam=="maxBins":
        indexList=maxBinsList
    df = pd.DataFrame(data=metrics,index=indexList,columns=["AUC","duration","impurty","maxDepth","maxBins","model"])
    showChart(df,"impurty","AUC","duration",0.5,0.7)

#展示图表
def showChart(df,evalParam,barData,lineData,yMin,yMax):
    """
    :param df: metrics 产生的dataframe
    :param evalParam: 此次评估的参数 例如 impurity
    :param barData: 绘出batchart数据 例如 AUC
    :param lineData: 绘出linedata数据，这里是duration
    :param yMin: y轴
    :param yMax:
    :return:
    """
    #绘制直方图
    ax=df[barData].plot(kind="bar",title=evalParam,flgsize=(10,6),legend=True,fontsize=12)
    ax.set_xlabel(evalParam,fontsize=12)
    ax.set_ylim(yMin,yMax)
    #绘制折线图
    ax2=ax.twinx()
    ax2.plot(df[lineData].values,linestyle="-",marker="o",linewidth=2.0,color="r")
    plt.show()


#评估模型
def trainEvaluateModel(trainData,validationData,impurtyParam,maxDepthParam,maxBinsParam):
    starttime=time()
    model=DecisionTree.trainClassifier(data=trainData,numClasses=2,categoricalFeaturesInfo={},impurity=impurtyParam,maxDepth=maxDepthParam,maxBins=maxBinsParam)
    AUC=evaluateModel(model,validationData)
    duration=time()-starttime
    print("训练评估使用参数：\n","impurity=",impurtyParam,"\n maxDepth=",maxDepthParam,"\n maxBins=",maxBinsParam,"====>用时=",duration,"\n 结果AUC=",AUC)
    return (AUC,duration,impurtyParam,maxDepthParam,maxBinsParam,model)


#评价模型计算AUC
def evaluateModel(model ,validationData):
    # 计算AUC（ROC曲线下的面积）
    score = model.predict(validationData.map(lambda x: x.features))
    print(score)
    scoreAndLabels = score.zip(validationData.map(lambda x: x.label))
    print("scoreAndLabels的前5项", scoreAndLabels.take(5))
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUC = metrics.areaUnderROC
    return (AUC)

#预测
def PredictData(sc,model,categoriesMap):
    print("开始导入数据")
    rawDataWithHeader=sc.textFile(Path+"test.tsv",minPartitions=20)
    header=rawDataWithHeader.first()
    rawData=rawDataWithHeader.filter(lambda x:x!=header)
    rData=rawData.map(lambda x:x.replace("\"",""))
    lines=rData.map(lambda x:x.split("\t"))
    print("总共有:", str(lines.count()))
    dataRDD=lines.map(lambda r:(r[0],extractFeatures(r,categoriesMap,len(r))))
    DescDict={
        0:"暂时性网页（ephemeral）",
        1:"长青网页（evergreen）"
    }
    for data in dataRDD.take(10):
        predictResult=model.predict(data[1])
        print("网址：",str(data[0]),"===》预测：",str(predictResult),"说明：",DescDict(predictResult))


# 数据转换，将文件的问号转换为0
def converFloat(x):
    return (0 if x == "?" else float(x))


# 返回特征字段
def extract_label(field):
    label = field[-1]
    return float(label)


# 提取
def extractFeatures(field, categoriesMap, featureEnd):
    # 提取分类特征字段
    categoryIdx = categoriesMap[field[3]]  # field[3]取到的是名字，然后categoriesMap取到类别编号
    categoryFeatures = np.zeros(len(categoriesMap))  # 创建categoriesMap的同型0矩阵
    categoryFeatures[categoryIdx] = 1  # 类别置1
    # 提取数值字段
    numericalFeatures = [converFloat(field) for field in field[4:featureEnd]]
    # 返回“分类特征字段”+“数值特征字段”
    result = np.concatenate((categoryFeatures, numericalFeatures))
    print(result)
    return result


# TODO 建立训练评估所需的数据，决策树训练需提供LabeldPoint格式的数据，LabeldPoint是由label与feature组成
def getLabeledRDD(lines, categoriesMap):
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extractFeatures(r, categoriesMap, len(r) - 1)))
    return labelpointRDD


# 将数据分为3部分，训练集，验证集合测试集
def get3PartData(labelpointRDD):
    # 数据集划分比例8:1:1
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print("数据集划分为：trainData:", str(trainData.count()), "validationData:", str(validationData.count()), "testData:",
          str(testData.count()))


def func1():
    data = getData()
    print("前两个：", data.take(2))


if __name__ == "__main__":
    getData()
