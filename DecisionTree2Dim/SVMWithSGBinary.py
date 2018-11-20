from pyspark import SparkContext
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from time import time
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.feature import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.mllib.classification import SVMWithSGD
"""
SVM二元分类
"""
sc = SparkContext('local')
global Path
Path = "D:\\workingSpace\\pythonAndSpark2\data\\stumbleupon\\"

#使用逻辑回归，数据需要归一化
def PrepareData(sc):
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
    #----2。创建训练所需的RDD数据
    categoriesMap = lines.map(lambda fields: fields[3]).distinct().zipWithIndex().collectAsMap()
    labelRDD=lines.map(lambda r:extract_label(r))
    featureRDD=lines.map(lambda r:extractFeatures(r, categoriesMap, len(r) - 1))
    print(featureRDD.first())
    #----3.随机分成3部分数据返回
    print("数据标准化之后===：")
    stdScaler=StandardScaler(withMean=True,withStd=True).fit(featureRDD)
    scalerFeatureRDD=stdScaler.transform(featureRDD)
    print(scalerFeatureRDD.first())
    labelPoint=labelRDD.zip(scalerFeatureRDD)
    labelpointRDD=labelPoint.map(lambda r:LabeledPoint(r[0],r[1]))
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print("数据集划分为：trainData:", str(trainData.count()), "validationData:", str(validationData.count()), "testData:",
          str(testData.count()))
    return (trainData, validationData, testData,categoriesMap)
# 数据转换，将文件的问号转换为0
def converFloat(x):
    return (float(0.0) if x == "?" else float(x))


# 返回特征字段
def extract_label(field):
    label = field[-1]
    return float(label)


# 提取
def extractFeatures(field, categoriesMap, featureEnd):
    # 提取分类特征字段
    categoryIdx = categoriesMap[field[3]]  # field[3]取到的是名字，然后categoriesMap取到类别编号
    categoryFeatures = np.zeros(len(categoriesMap))  # 创建categoriesMap的同型0矩阵
    categoryFeatures[categoryIdx] = float(1.0)  # 类别置1
    # 提取数值字段
    numericalFeatures = [converFloat(field) for field in field[4:featureEnd]]
    # 返回“分类特征字段”+“数值特征字段”
    result = np.concatenate((categoryFeatures, numericalFeatures))
    print(result)
    return result
#评估模型
def trainEvaluateModel(trainData,validationData,numIterations,stepSize,regParam):
    """
    :param trainData: 训练集
    :param validationData: 验证集
    :param numIterations: 迭代次数
    :param stepSize: 步长
    :param regParam: 正则化参数
    :return:
    """
    starttime=time()
    model=SVMWithSGD.train(data=trainData,iterations=numIterations,step=stepSize,regParam=regParam)
    AUC=evaluateModel(model,validationData)
    duration=time()-starttime
    print("训练评估使用参数：\n","numIterations=",numIterations,"\n stepSize=",stepSize,"\n regParam=",regParam,"====>用时=",duration,"\n 结果AUC=",AUC)
    return (AUC,duration,numIterations,stepSize,regParam,model)

def parameterEval(trainData,validationData):
    print("评估numIeteration参数===========")
    evalParammeter(trainData=trainData,
                   validationData=validationData,
                   evalParam="numIeteration",
                   numIterationList=[5,10,15,20,60,100],
                   stepSizeList=[10],
                   regParamList=[1])
    print("评估stepSize参数===========")
    evalParammeter(trainData=trainData,
                   validationData=validationData,
                   evalParam="stepSize",
                   numIterationList=[50],
                   stepSizeList=[10,50,100,200],
                   regParamList=[1])
    print("评估regParam参数===========")
    evalParammeter(trainData=trainData,
                   validationData=validationData,
                   evalParam="regParam",
                   numIterationList=[50],
                   stepSizeList=[500],
                   regParamList=[0.01,0.1,1])


#评估摸一个参数
def evalParammeter(trainData,validationData,evalParam,numIterationList,stepSizeList,regParamList):
    # 训练评估参数
    metrics = [trainEvaluateModel(trainData, validationData, numIteration, stepSize, regParam)
               for numIteration in numIterationList
               for stepSize in stepSizeList
               for regParam in regParamList]
    # 设置当前评估的参数
    if evalParam == "numIteration":
        indexList = numIterationList
    elif evalParam == "stepSize":
        indexList = stepSizeList
    elif evalParam == "regParam":
        indexList = regParamList
    df = pd.DataFrame(data=metrics, index=indexList,
                      columns=["AUC", "duration", "numIteration", "stepSize", "regParam", "model"])
    showChart(df, evalParam, "AUC", "duration", 0.5, 0.7)

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
    # 绘制直方图
    ax = df[barData].plot(kind="bar", title=evalParam, figsize=(10, 6), legend=True, fontsize=12)
    ax.set_xlabel(evalParam, fontsize=12)
    ax.set_ylim([yMin, yMax])
    ax.set_ylabel(barData, fontsize=12)
    # 绘制折线图
    ax2 = ax.twinx()
    ax2.plot(df[[lineData]].values, linestyle="-", marker="o", linewidth=2.0, color="r")
    plt.show()
#评价模型计算AUC
def evaluateModel(model ,validationData):
    # 计算AUC（ROC曲线下的面积）
    score = model.predict(validationData.map(lambda x: x.features))
    print(score)
    #将整数int转换为float，下面的方法需要float类型数据，否则metrics.areaUnderROC报错
    score = score.map(lambda x: x + 0.0)
    scoreAndLabels = score.zip(validationData.map(lambda x: x.label))
    print("scoreAndLabels的前5项", scoreAndLabels.take(5))
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUC = metrics.areaUnderROC
    return (AUC)
#评估所有参数，选择最佳参数组合
def evalAllParammeter(trainData,validationData,numIterationList,stepSizeList,regParamList):
    #for循环遍历所有参数集合
    metrics = [trainEvaluateModel(trainData, validationData, numIterations,stepSize,regParam)
                for numIterations in numIterationList
                for stepSize in stepSizeList
                for regParam in regParamList]
    #找出AUC最大的参数组合
    Smetrics=sorted(metrics,key=lambda k:k[0],reverse=True)
    bestParammeter=Smetrics[0]
    #显示调校后的最佳参数组合
    print("调教的最佳参数: numIterations=",bestParammeter[2],",stepSize=",bestParammeter[3],",regParam=",bestParammeter[4],"结果AUC=",bestParammeter[0])
    #返回最佳模型
    return bestParammeter[5]
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
        print("网址：",str(data[0]),"===》预测：",str(predictResult),"说明：",DescDict[predictResult])

if __name__=="__main__":
    print("程序启动。。。。。。。。。。")
    print("数据准备金阶段==================")
    (trainData,validationData,testData,categoriesMap)=PrepareData(sc)
    # 将数据暂存内存中
    trainData.persist()
    validationData.persist()
    testData.persist()
    print("训练评估阶段===================")
    (AUC,duration,numIterations,stepSize,miniBatchFraction,model)=trainEvaluateModel(trainData,validationData,3,50,1)
    print("找出最好的参数组合=============")
    model=evalAllParammeter(trainData=trainData,
                            validationData=validationData,
                            numIterationList=[1],
                            stepSizeList=[10],
                            regParamList=[0.01]
                            # numIterationList=[1,3,5,15,25],
                            # stepSizeList=[10,50,100],
                            #  regParamList=[0.01,0.1,1]
                        )
    print("测试阶段====================")
    auc=evaluateModel(model,testData)
    print("使用最佳模型，结果AUc:",auc)
    print("预测数据：\n")
    PredictData(sc,model,categoriesMap)
    # print(model.toDebugString())



























