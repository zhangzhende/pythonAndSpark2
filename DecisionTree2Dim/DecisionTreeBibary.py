from pyspark import SparkContext
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from time import time
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import BinaryClassificationMetrics

"""
决策树二元分类
"""
sc = SparkContext('local')
global Path
Path = "D:\\workingSpace\\pythonAndSpark2\data\\stumbleupon\\"


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
    # ----2。创建训练所需的RDD数据
    #字典{busi:1}这样
    categoriesMap = lines.map(lambda fields: fields[3]).distinct().zipWithIndex().collectAsMap()
    labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extractFeatures(r, categoriesMap, len(r) - 1)))
    # ----3.随机分成3部分数据返回
    (trainData, validationData, testData) = labelpointRDD.randomSplit([8, 1, 1])
    print("数据集划分为：trainData:", str(trainData.count()), "validationData:", str(validationData.count()), "testData:",
          str(testData.count()))
    return (trainData, validationData, testData, categoriesMap)


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
    categoryFeatures[categoryIdx] = 1  # 类别置1，onehot编码
    # 提取数值字段
    numericalFeatures = [converFloat(field) for field in field[4:featureEnd]]
    # 返回“分类特征字段”+“数值特征字段”
    result = np.concatenate((categoryFeatures, numericalFeatures))
    print(result)
    return result


# 评估模型
def trainEvaluateModel(trainData, validationData, impurtyParam, maxDepthParam, maxBinsParam):
    starttime = time()
    model = DecisionTree.trainClassifier(data=trainData, numClasses=2, categoricalFeaturesInfo={},
                                         impurity=impurtyParam, maxDepth=maxDepthParam, maxBins=maxBinsParam)
    AUC = evaluateModel(model, validationData)
    duration = time() - starttime
    print("训练评估使用参数：\n", "impurity=", impurtyParam, "\n maxDepth=", maxDepthParam, "\n maxBins=", maxBinsParam,
          "====>用时=", duration, "\n 结果AUC=", AUC)
    return (AUC, duration, impurtyParam, maxDepthParam, maxBinsParam, model)


# 评价模型计算AUC
def evaluateModel(model, validationData):
    # 计算AUC（ROC曲线下的面积）
    score = model.predict(validationData.map(lambda x: x.features))
    print(score)
    scoreAndLabels = score.zip(validationData.map(lambda x: x.label))
    print("scoreAndLabels的前5项", scoreAndLabels.take(5))
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    AUC = metrics.areaUnderROC
    return (AUC)


# 评估所有参数，选择最佳参数组合
def evalAllParammeter(trainData, validationData, impurityList, maxDepthList, maxBinsList):
    # for循环遍历所有参数集合
    metrics = [trainEvaluateModel(trainData, validationData, impurty, maxDepth, maxBins)
               for impurty in impurityList
               for maxDepth in maxDepthList
               for maxBins in maxBinsList]
    # 找出AUC最大的参数组合
    Smetrics = sorted(metrics, key=lambda k: k[0], reverse=True)
    bestParammeter = Smetrics[0]
    # 显示调校后的最佳参数组合
    print("调教的最佳参数: impurity=", bestParammeter[2], ",maxDepth=", bestParammeter[3], ",maxBins=", bestParammeter[4],
          "结果AUC=", bestParammeter[0])
    # 返回最佳模型
    return bestParammeter[5]


# 预测
def PredictData(sc, model, categoriesMap):
    print("开始导入数据")
    rawDataWithHeader = sc.textFile(Path + "test.tsv", minPartitions=20)
    header = rawDataWithHeader.first()
    rawData = rawDataWithHeader.filter(lambda x: x != header)
    rData = rawData.map(lambda x: x.replace("\"", ""))
    lines = rData.map(lambda x: x.split("\t"))
    print("总共有:", str(lines.count()))
    dataRDD = lines.map(lambda r: (r[0], extractFeatures(r, categoriesMap, len(r))))
    DescDict = {
        0: "暂时性网页（ephemeral）",
        1: "长青网页（evergreen）"
    }
    for data in dataRDD.take(10):
        predictResult = model.predict(data[1])
        print("网址：", str(data[0]), "===》预测：", str(predictResult), "说明：", DescDict[predictResult])


if __name__ == "__main__":
    print("程序启动。。。。。。。。。。")
    print("数据准备金阶段==================")
    (trainData, validationData, testData, categoriesMap) = PrepareData(sc)
    # 将数据暂存内存中
    trainData.persist()
    validationData.persist()
    testData.persist()
    print("训练评估阶段===================")
    (AUC, duration, impurityParam, maxDepthParam, MaxBinsParam, model) = trainEvaluateModel(trainData, validationData,
                                                                                            "entropy", 5, 5)
    print("找出最好的参数组合=============")
    model = evalAllParammeter(trainData=trainData,
                              validationData=validationData,
                              impurityList=["gini", "entropy"],
                              maxDepthList=[3, 5, 10, 15, 20, 25],
                              maxBinsList=[3, 5, 10, 50, 100, 200])
    print("测试阶段====================")
    auc = evaluateModel(model, testData)
    print("使用最佳模型，结果AUc:", auc)
    print("预测数据：\n")
    PredictData(sc, model, categoriesMap)
    print(model.toDebugString())
