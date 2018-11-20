from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from time import time
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics
import pandas as pd
import matplotlib.pyplot as plt

"""
决策树多元分类
"""
sc = SparkContext('local')
global Path
Path = "D:\\workingSpace\\pythonAndSpark2\data\\covtype\\"


def PrepareData(sc):
    print("开始导入数据。。。")
    path = Path + "covtype.data"
    print(path)
    # 使用minPartitions=40，将数据分成40片，不然报错
    rawData = sc.textFile(path, minPartitions=40)
    header = rawData.first()
    # print(header)
    # 按照逗号符分字段
    lines = rawData.map(lambda x: x.split(","))
    print("总共有:", str(lines.count()))

    # ----2。创建训练所需的RDD数据
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extractFeatures(r, len(r) - 1)))
    # ----3.随机分成3部分数据返回
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print("数据集划分为：trainData:", str(trainData.count()), "validationData:", str(validationData.count()), "testData:",
          str(testData.count()))
    print("labelpointRDD.first():", labelpointRDD.first())
    return (trainData, validationData, testData)


# 数据转换，将文件的问号转换为0
def converFloat(x):
    return (float(0.0) if x == "?" else float(x))


# 返回特征字段,
def extract_label(record):
    label = record[-1]
    # 下面减一是为了让下标从0开始，原始数据的下标从1开始对应的
    return float(label) - 1.0


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
                      columns=["accuracy", "duration", "impurity", "maxDepth", "maxBins", "model"])
    showChart(df, evalParam, "accuracy", "duration", 0.6, 1.0)


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


# 提取特征
def extractFeatures(record, featureEnd):
    # 提取数值字段
    numericalFeatures = [converFloat(field) for field in record[0:featureEnd]]
    # print(numericalFeatures)
    return numericalFeatures


# 评估模型
def trainEvaluateModel(trainData, validationData, impurtyParam, maxDepthParam, maxBinsParam):
    starttime = time()
    model = DecisionTree.trainClassifier(data=trainData, numClasses=7, categoricalFeaturesInfo={},
                                         impurity=impurtyParam, maxDepth=maxDepthParam, maxBins=maxBinsParam)
    accuracy = evaluateModel(model, validationData)
    duration = time() - starttime
    print("训练评估使用参数：\n", "impurity=", impurtyParam, "\n maxDepth=", maxDepthParam, "\n maxBins=", maxBinsParam,
          "====>用时=", duration, "\n 结果accuracy=", accuracy)
    return (accuracy, duration, impurtyParam, maxDepthParam, maxBinsParam, model)


# 评价模型计算accuracy
def evaluateModel(model, validationData):
    # 计算accuracy
    score = model.predict(validationData.map(lambda x: x.features))
    # print(score)
    score = score.map(lambda x: float(x))
    scoreAndLabels = score.zip(validationData.map(lambda x: float(x.label)))
    print("scoreAndLabels的前2项", scoreAndLabels.take(2))
    metrics = MulticlassMetrics(scoreAndLabels)
    accuracy = metrics.accuracy
    return (accuracy)


# 评估所有参数，选择最佳参数组合
def evalAllParammeter(trainData, validationData, impurityList, maxDepthList, maxBinsList):
    # for循环遍历所有参数集合
    metrics = [trainEvaluateModel(trainData, validationData, impurty, maxDepth, maxBins)
               for impurty in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]
    # 找出accuracy最大的参数组合
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParammeter = Smetrics[0]
    # 显示调校后的最佳参数组合
    print("调教的最佳参数: impurity=", bestParammeter[2], ",maxDepth=", bestParammeter[3], ",maxBins=", bestParammeter[4],
          "结果accuracy=", bestParammeter[0])
    # 返回最佳模型
    return bestParammeter[5]


# 预测,还是拿原始数据预测
def PredictData(sc, model):
    print("开始导入数据")
    rawData = sc.textFile(Path + "covtype.data", minPartitions=20)
    lines = rawData.map(lambda x: x.split(","))
    print("总共有:", str(lines.count()))
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extractFeatures(r, len(r) - 1)))
    for data in labelpointRDD.take(10):
        predictResult = model.predict(data.features)
        label = data.label
        features = data.features
        result = ("正确" if (label == predictResult) else "错误")
        print("土地条件：海拔:" + str(features[0]) +
              " 方位:" + str(features[1]) +
              " 斜率:" + str(features[2]) +
              " 水源垂直距离:" + str(features[3]) +
              " 水源水平距离:" + str(features[4]) +
              " 9点时阴影:" + str(features[5]) +
              "....==>预测:" + str(predictResult) +
              " 实际:" + str(label) + "结果:" + result)


if __name__ == "__main__":
    print("程序启动。。。。。。。。。。")
    print("数据准备金阶段==================")
    (trainData, validationData, testData) = PrepareData(sc)
    # 将数据暂存内存中
    trainData.persist()
    validationData.persist()
    testData.persist()
    print("训练评估阶段===================")
    (accuracy, duration, impurityParam, maxDepthParam, MaxBinsParam, model) = trainEvaluateModel(trainData,
                                                                                                 validationData,
                                                                                                 "entropy", 5, 5)
    print("找出最好的参数组合=============")
    model = evalAllParammeter(trainData=trainData,
                              validationData=validationData,
                              impurityList=["entropy"],
                              maxDepthList=[3],
                              maxBinsList=[3])
                              # impurityList=["gini", "entropy"],
                              # maxDepthList=[3, 5, 10, 15, 20, 25],
                              # maxBinsList=[3, 5, 10, 50, 100, 200])
    print("测试阶段====================")
    auc = evaluateModel(model, testData)
    print("使用最佳模型，结果accuracy:", accuracy)
    print("预测数据：\n")
    PredictData(sc, model)
    print(model.toDebugString())
