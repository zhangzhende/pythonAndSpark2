from pyspark import SparkContext
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from time import time
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import RegressionMetrics
import pandas as pd
import matplotlib.pyplot as plt
import math

"""
决策树回归分析
"""
sc = SparkContext('local')
global Path
Path = "D:\\workingSpace\\pythonAndSpark2\data\\Bike\\"


def PrepareData(sc):
    print("开始导入数据。。。")
    path = Path + "hour.csv"
    print(path)
    # 使用minPartitions=40，将数据分成40片，不然报错
    rawDataWithHeader = sc.textFile(path, minPartitions=40)
    header = rawDataWithHeader.first()
    # 去掉首行，标题
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    # 按照，分字段
    lines = rawData.map(lambda x: x.split(","))
    print("总共有:", str(lines.count()))
    # ----2。创建训练所需的RDD数据
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extractFeatures(r, len(r) - 1)))
    # ----3.随机分成3部分数据返回
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print("数据集划分为：trainData:", str(trainData.count()), "validationData:", str(validationData.count()), "testData:",
          str(testData.count()))
    return (trainData, validationData, testData)


# 数据转换，将文件的问号转换为0
def converFloat(x):
    return (0 if x == "?" else float(x))


# 返回特征字段
def extract_label(field):
    label = (field[-1])
    return float(label)


# 提取
def extractFeatures(record, featureEnd):
    # 提取数值字段
    featureSeason = [converFloat(field) for field in record[2]]
    features=[converFloat(field) for field in record[4:featureEnd-2]]
    result=np.concatenate((featureSeason,features))
    return result


# 评估模型
def trainEvaluateModel(trainData, validationData, impurtyParam, maxDepthParam, maxBinsParam):
    starttime = time()
    model = DecisionTree.trainRegressor(data=trainData, categoricalFeaturesInfo={},
                                         impurity=impurtyParam, maxDepth=maxDepthParam, maxBins=maxBinsParam)
    RMSE = evaluateModel(model, validationData)#均方根误差
    duration = time() - starttime
    print("训练评估使用参数：\n", "impurity=", impurtyParam, "\n maxDepth=", maxDepthParam, "\n maxBins=", maxBinsParam,
          "====>用时=", duration, "\n 结果AUC=", RMSE)
    return (RMSE, duration, impurtyParam, maxDepthParam, maxBinsParam, model)


# 评价模型计算AUC
def evaluateModel(model, validationData):
    # 计算AUC（ROC曲线下的面积）
    score = model.predict(validationData.map(lambda x: x.features))
    print(score)
    scoreAndLabels = score.zip(validationData.map(lambda x: x.label))
    print("scoreAndLabels的前5项", scoreAndLabels.take(5))
    metrics = RegressionMetrics(scoreAndLabels)
    RMSE = metrics.rootMeanSquaredError
    return (RMSE)


# 评估所有参数，选择最佳参数组合
def evalAllParammeter(trainData, validationData, impurityList, maxDepthList, maxBinsList):
    # for循环遍历所有参数集合
    metrics = [trainEvaluateModel(trainData, validationData, impurty, maxDepth, maxBins)
               for impurty in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]
    # 找出均方根误差最小的
    Smetrics = sorted(metrics, key=lambda k: k[0])
    bestParammeter = Smetrics[0]
    # 显示调校后的最佳参数组合
    print("调教的最佳参数: impurity=", bestParammeter[2], ",maxDepth=", bestParammeter[3], ",maxBins=", bestParammeter[4],
          "结果RMSE=", bestParammeter[0])
    # 返回最佳模型
    return bestParammeter[5]
# 评估摸一个参数
def evalParammeter(trainData, validationData, evalParam, impurityList, maxDepthList, maxBinsList):
    # 训练评估参数
    metrics = [trainEvaluateModel(trainData, validationData, impurity, maxDepth, maxBins)
               for impurity in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]
    # 设置当前评估的参数
    if evalParam == "impurity":
        indexList = impurityList
    elif evalParam == "maxDepth":
        indexList = maxDepthList
    elif evalParam == "maxBins":
        indexList = maxBinsList
    df = pd.DataFrame(data=metrics, index=indexList,
                      columns=["RMSE", "duration", "impurity", "maxDepth", "maxBins", "model"])
    showChart(df, evalParam, "RMSE", "duration", 0, 200)


def showChart(df, evalParam, barData, lineData, yMin, yMax):
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

def parameterEval(trainData, validationData):
    print("评估maxDepth参数===========")
    evalParammeter(trainData=trainData,
                   validationData=validationData,
                   evalParam="maxDepth",
                   impurityList=["variance"],
                   maxDepthList=[3,5,10,15,20,25],
                   maxBinsList=[10])
    print("评估maxBins参数===========")
    evalParammeter(trainData=trainData,
                   validationData=validationData,
                   evalParam="maxBins",
                   impurityList=["variance"],
                   maxDepthList=[10],
                   maxBinsList=[3,5,10,50,100])

# 预测
def PredictData(sc, model):
    print("开始导入数据")
    path = Path + "hour.csv"
    print(path)
    # 使用minPartitions=40，将数据分成40片，不然报错
    rawDataWithHeader = sc.textFile(path, minPartitions=40)
    header = rawDataWithHeader.first()
    # 去掉首行，标题
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    # 按照，分字段
    lines = rawData.map(lambda x: x.split(","))
    print("总共有:", str(lines.count()))
    # ----2。创建训练所需的RDD数据
    dataRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extractFeatures(r, len(r) - 1)))
    #--------------3定义字典
    SeasonDict = { 1: "春", 2: "夏",3:"秋",4:"冬" }
    HolidayDict={0:"非假日",1:"假日"}
    WeekDict={0:"一",1:"二",2:"三",3:"四",4:"五",5:"六",6:"日"}
    WorkDay={0:"非工作日",1:"工作日"}
    WeatherDict={1:"晴",2:"阴",3:"小雨",4:"大雨"}
    for data in dataRDD.take(10):
        predictResult = int(model.predict(data.features))
        label = data.label
        features = data.features
        result = ("正确" if (label == predictResult) else "错误")
        error=math.fabs(label-predictResult)
        dataDesc="特征："+SeasonDict[features[0]]+"季节，"+str(features[1])+"月，"+str(features[2])+"时，" \
        +HolidayDict[features[3]]+",星期"+WeatherDict[features[4]]+","+WorkDay[features[5]]+","+WeatherDict[features[6]] \
        +","+str(features[7]*41)+"度，体感"+str(features[8]*50)+"度,湿度"+str(features[9]*100)+",风速"+str(features[10]*67)+",预测结果===》" \
        +str(predictResult)+"，实际："+str(label)+result+"误差："+str(error)
        print(dataDesc)


if __name__ == "__main__":
    print("程序启动。。。。。。。。。。")
    print("数据准备金阶段==================")
    (trainData, validationData, testData) = PrepareData(sc)
    # 将数据暂存内存中
    trainData.persist()
    validationData.persist()
    testData.persist()
    print("训练评估阶段===================")
    (RMSE, duration, impurityParam, maxDepthParam, MaxBinsParam, model) = trainEvaluateModel(trainData, validationData,
                                                                                            "variance", 5, 5)
    print("找出最好的参数组合=============")
    model = evalAllParammeter(trainData=trainData,
                              validationData=validationData,
                              impurityList=["variance"],
                              maxDepthList=[3, 5, 10, 15, 20, 25],
                              maxBinsList=[3, 5, 10, 50, 100, 200])
    print("测试阶段====================")
    auc = evaluateModel(model, testData)
    print("使用最佳模型，结果RMSE:", RMSE)
    print("预测数据：\n")
    PredictData(sc, model)
    print(model.toDebugString())
