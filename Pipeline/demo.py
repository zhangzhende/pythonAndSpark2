from pyspark import SparkContext
from pyspark.sql import Row, SparkSession
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import udf,col
import pyspark.sql.types
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier,RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder,TrainValidationSplit,CrossValidator


"""
pipeline机器学习流程二元分类使用
"""
sc = SparkContext('local')
Path = "D:\\workingSpace\\pythonAndSpark2\data\\ml-100k\\"
sqlContext = SparkSession.builder.getOrCreate()


def func():
    row_df = sqlContext.read.format("csv").option("header", True).option("delimiter", "\t").load(Path + "train.tsv")
    print("row_df.count()打印项数：", row_df.count())
    row_df.printSchema()
    row_df.show(5)
    df= row_df.select(["url","alchemy_category"]#不需要转换的字段
                      +[replace_question(col(column)).cast("double").alias(column) for column in row_df.columns[4:]])#需要转换的字段
    print(df.printSchema())
    df.show(3)
    train_df,test_df=df.randomSplit([0.7,0.3])
    train_df.cache()
    test_df.cache()

    categoryIndexer=StringIndexer(inputCol="alchemy_category",outputCol="alchemy_category_index")#创建indexer,字符串代码化
    categoryTransformer=categoryIndexer.fit(df)#生成transformer
    for i in range(0,len(categoryTransformer.labels)):#查看
        print(str(i),categoryTransformer.labels[i])
    df1=categoryTransformer.transform(train_df)#train_df转换为df1
    print(df1.columns)
    df1.show(5)

    #onehotencode
    encoder=OneHotEncoder(dropLast=False,inputCol="alchemy_category_index",outputCol="alchemy_category_indexVec")
    df2=encoder.transform(df1)
    print(df2.columns)

    #vectorAssembler
    assemblerInputs=["alchemy_category_indexVec"]+row_df.columns[4:-1]
    print(assemblerInputs)
    assembler=VectorAssembler(inputCols=assemblerInputs,outputCol="features")
    df3=assembler.transform(df2)
    print(df3.columns)
    df3.select("features").show(5)
    print(df3.select("features").take(1))

    ####使用决策树二元分类
    dt=DecisionTreeClassifier(labelCol="label",featuresCol="features",impurity="gini",maxDepth=10,maxBins=14)
    dt_model=dt.fit(df3)
    print(dt_model)
    df4=dt_model.transform(df3)

def func2():
    """
    PIPeline机器学习
    :return:
    """
    row_df = sqlContext.read.format("csv").option("header", True).option("delimiter", "\t").load(Path + "train.tsv")
    df = row_df.select(["url", "alchemy_category"]  # 不需要转换的字段
                       + [replace_question(col(column)).cast("double").alias(column) for column in
                          row_df.columns[4:]])  # 需要转换的字段
    train_df, test_df = df.randomSplit([0.7, 0.3])
    ###建立机器学习Pipeline流程
    stringIndexer = StringIndexer(inputCol="alchemy_category", outputCol="alchemy_category_index")  # 创建indexer,字符串代码化
    encoder=OneHotEncoder(dropLast=False,inputCol="alchemy_category_index",outputCol="alchemy_category_indexVec")
    assemblerInputs = ["alchemy_category_indexVec"] + row_df.columns[4:-1]
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", impurity="gini", maxDepth=10, maxBins=14)
    pipeline=Pipeline(stages=[stringIndexer,encoder,assembler,dt])
    print(pipeline.getStages())

    ###使用Pipeline进行数据处理和训练
    pipelineModel=pipeline.fit(train_df)#训练
    print(pipelineModel.stages[3])#第三阶段会产生模型，这里看看模型
    print(pipelineModel.stages[3].toDebugString)

    ####使用pipeline进行预测
    predicted=pipelineModel.transform(test_df)
    print(predicted.columns)
    predicted.select("url","features","rawprediction","probability","label","prediction").show(5)
    predicted.select( "probability",  "prediction").take(5)

    ####评估模型准确率
    evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol="label",metricName="areaUnderROC")
    auc=evaluator.evaluate(predicted)
    print("auc:",auc)

    #要遍历的参数们，选择最佳参数组合
    paramGrid=ParamGridBuilder().addGrid(dt.impurity,["gini","entory"]).addGrid(dt.maxDepth,[5,10,15]).addGrid(dt.maxBins,[10,15,20]).build()
    tvs=TrainValidationSplit(estimator=dt,evaluator=evaluator,estimatorParamMaps=paramGrid,trainRatio=0.8)#trainRatio 数据会8:2的比例分为训练集，验证集
    tvs_pipeline=Pipeline(stages=[stringIndexer,encoder,assembler,tvs])
    tvs_pipelineModel=tvs_pipeline.fit(train_df)
    bestModel=tvs_pipelineModel.stages[3].bestModel
    print("bestModel",bestModel)
    predictions=tvs_pipelineModel.transform(test_df)
    auc2=evaluator.evaluate(predictions)
    print("auc2:",auc2)

def func3():
    """
    使用交叉验证
    :return:
    """
    row_df = sqlContext.read.format("csv").option("header", True).option("delimiter", "\t").load(Path + "train.tsv")
    df = row_df.select(["url", "alchemy_category"]  # 不需要转换的字段
                       + [replace_question(col(column)).cast("double").alias(column) for column in
                          row_df.columns[4:]])  # 需要转换的字段
    train_df, test_df = df.randomSplit([0.7, 0.3])
    ###建立机器学习Pipeline流程
    stringIndexer = StringIndexer(inputCol="alchemy_category", outputCol="alchemy_category_index")  # 创建indexer,字符串代码化
    encoder = OneHotEncoder(dropLast=False, inputCol="alchemy_category_index", outputCol="alchemy_category_indexVec")
    assemblerInputs = ["alchemy_category_indexVec"] + row_df.columns[4:-1]
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", impurity="gini", maxDepth=10, maxBins=14)
    # pipeline = Pipeline(stages=[stringIndexer, encoder, assembler, dt])
    # print(pipeline.getStages())

    ###使用Pipeline进行数据处理和训练
    # pipelineModel = pipeline.fit(train_df)  # 训练
    # print(pipelineModel.stages[3])  # 第三阶段会产生模型，这里看看模型
    # print(pipelineModel.stages[3].toDebugString)

    ####使用pipeline进行预测
    # predicted = pipelineModel.transform(test_df)
    # print(predicted.columns)
    # predicted.select("url", "features", "rawprediction", "probability", "label", "prediction").show(5)
    # predicted.select("probability", "prediction").take(5)

    ####评估模型准确率
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label",
                                              metricName="areaUnderROC")
    # auc = evaluator.evaluate(predicted)
    # print("auc:", auc)

    # 要遍历的参数们，选择最佳参数组合
    paramGrid = ParamGridBuilder().addGrid(dt.impurity, ["gini", "entory"]).addGrid(dt.maxDepth, [5, 10, 15]).addGrid(
        dt.maxBins, [10, 15, 20]).build()
    #3折交叉验证
    cv=CrossValidator(estimator=dt, evaluator=evaluator, estimatorParamMaps=paramGrid,numFolds=3)
    cv_pipeline = Pipeline(stages=[stringIndexer, encoder, assembler, cv])
    cv_pipelineModel = cv_pipeline.fit(train_df)
    bestModel = cv_pipelineModel.stages[3].bestModel
    predictions = cv_pipelineModel.transform(test_df)
    auc2 = evaluator.evaluate(predictions)
    print("bestModel", bestModel)
    print("auc2:", auc2)

def func4():
    """
    使用随机森林
    :return:
    """
    row_df = sqlContext.read.format("csv").option("header", True).option("delimiter", "\t").load(Path + "train.tsv")
    df = row_df.select(["url", "alchemy_category"]  # 不需要转换的字段
                       + [replace_question(col(column)).cast("double").alias(column) for column in
                          row_df.columns[4:]])  # 需要转换的字段
    train_df, test_df = df.randomSplit([0.7, 0.3])
    ###建立机器学习Pipeline流程
    stringIndexer = StringIndexer(inputCol="alchemy_category", outputCol="alchemy_category_index")  # 创建indexer,字符串代码化
    encoder = OneHotEncoder(dropLast=False, inputCol="alchemy_category_index", outputCol="alchemy_category_indexVec")
    assemblerInputs = ["alchemy_category_indexVec"] + row_df.columns[4:-1]
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    # dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", impurity="gini", maxDepth=10, maxBins=14)
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)

    ####评估模型准确率
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label",
                                              metricName="areaUnderROC")
    # 要遍历的参数们，选择最佳参数组合
    paramGrid = ParamGridBuilder().addGrid(rf.impurity, ["gini", "entory"]).addGrid(rf.maxDepth, [5, 10, 15]).addGrid(
        rf.maxBins, [10, 15, 20]).addGrid(rf.numTrees,[10,20,30]).build()

    rf_pipeline = Pipeline(stages=[stringIndexer, encoder, assembler, rf])
    rf_pipelineModel = rf_pipeline.fit(train_df)
    predictions = rf_pipelineModel.transform(test_df)
    auc3 = evaluator.evaluate(predictions)
    #3折交叉验证
    cv=CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=paramGrid,numFolds=3)
    cv_pipeline = Pipeline(stages=[stringIndexer, encoder, assembler, cv])
    cv_pipelineModel = cv_pipeline.fit(train_df)
    bestModel = cv_pipelineModel.stages[3].bestModel
    print("bestModel", bestModel)
    predictions2 = cv_pipelineModel.transform(test_df)
    auc2 = evaluator.evaluate(predictions2)
    print("auc2:", auc2)
    DescDict = {
        0: "暂时性网页（ephemeral）",
        1: "长青网页（evergreen）"
    }
    for data in predictions2.select("url","prediction").take(5):
        print("网址：", str(data[0]), "===》预测：", str(data[1]), "说明：", DescDict[data[1]])


# dataFrames UDF 用户自定义函数
def replace_question(x):
    return ("0" if x == "?" else x)
replace_question = udf(replace_question)