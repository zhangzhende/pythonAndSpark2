from pyspark import SparkContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import StringType,StructType,StructField
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler,VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder,TrainValidationSplit,CrossValidator
from pyspark.ml.regression import DecisionTreeRegressor,GBTRegressor
"""
使用pipeline机器学习流程回归分析
"""
sc = SparkContext('local')
Path = "D:\\workingSpace\\pythonAndSpark2\data\\Bike\\"
sqlContext = SparkSession.builder.getOrCreate()

def func1():
    hour_df=sqlContext.read.format("csv").option("header","true").load(Path+"hour.csv")
    print("count",hour_df.count())
    print("columns:",hour_df.columns)
    #舍弃不需要的字段
    hour_df=hour_df.drop("instant").drop("dteday").drop("yr").drop("casual").drop("registered")
    print("查看schema:",hour_df.printSchema())
    # 数据转换为double
    hour_df = hour_df.select([col(column).cast("double").alias(column) for column in hour_df.columns])
    print("转换后：hour_df.printSchema()：", hour_df.printSchema())
    print("前3项数据：", hour_df.show(3))
    # 将数据分为train_df和test_df，比例为0.7:0.3
    train_df, test_df = hour_df.randomSplit([0.7, 0.3])
    train_df.cache()
    test_df.cache()
    # 创建特征字段list
    featureCols = hour_df.columns[:-1]
    print("featureCols:", featureCols)
    # 建立pipeline
    vectorAssembler = VectorAssembler(inputCols=featureCols, outputCol="aFeatures")
    vectorIndexer=VectorIndexer(inputCol="aFeatures",outputCol="features",maxCategories=24)
    dt=DecisionTreeRegressor(labelCol="cnt",featuresCol="features")
    dt_pipeline=Pipeline(stages=[vectorAssembler,vectorIndexer,dt])
    print("查看pipeline流程：", dt_pipeline.getStages())
    # 训练
    dt_pipelineModel = dt_pipeline.fit(dataset=train_df)
    print("查看训练完成后的模型：", dt_pipelineModel.stages[2].toDebugString[:500])
    # 使用transform预测
    predicted = dt_pipelineModel.transform(test_df)
    print("查看新增的字段：", predicted.columns)
    print("查看预测的结果：", predicted.show(2))
    ###评估模型
    evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="rmse")
    predicted_df=dt_pipelineModel.transform(test_df)
    rmse=evaluator.evaluate(predicted_df)
    print("rmse:",rmse)
    ##TrainValidationSplit训练找出最佳模型
    paramGrid = ParamGridBuilder().addGrid(dt.impurity, ["gini", "entory"]).addGrid(dt.maxDepth, [5, 10, 15]).addGrid(
        dt.maxBins, [10, 15, 20]).build()
    tvs = TrainValidationSplit(estimator=dt, evaluator=evaluator, estimatorParamMaps=paramGrid,
                               trainRatio=0.8)  # trainRatio 数据会8:2的比例分为训练集，验证集
    tvs_pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer,tvs])
    yvs_pipelineModel = tvs_pipeline.fit(dataset=train_df)
    bestmodel = yvs_pipelineModel.stages[2].bestModel
    print("bestModel:", bestmodel.toDebugString[:500])
    ##使用最佳模型进行预测
    predictions = tvs_pipeline.transform(test_df)
    rmse2=evaluator.evaluate(predictions)
    print(rmse2)

def func2():
    """
    使用K折交叉验证
    :return:
    """
    hour_df = sqlContext.read.format("csv").option("header", "true").load(Path + "hour.csv")
    # 舍弃不需要的字段
    hour_df = hour_df.drop("instant").drop("dteday").drop("yr").drop("casual").drop("registered")
    # 数据转换为double
    hour_df = hour_df.select([col(column).cast("double").alias(column) for column in hour_df.columns])
    # 将数据分为train_df和test_df，比例为0.7:0.3
    train_df, test_df = hour_df.randomSplit([0.7, 0.3])
    train_df.cache()
    test_df.cache()
    # 创建特征字段list
    featureCols = hour_df.columns[:-1]
    # 建立pipeline
    vectorAssembler = VectorAssembler(inputCols=featureCols, outputCol="aFeatures")
    vectorIndexer = VectorIndexer(inputCol="aFeatures", outputCol="features", maxCategories=24)
    dt = DecisionTreeRegressor(labelCol="cnt", featuresCol="features")
    dt_pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, dt])
    # 训练
    dt_pipelineModel = dt_pipeline.fit(dataset=train_df)
    # 使用transform预测
    predicted = dt_pipelineModel.transform(test_df)
    ###评估模型
    evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="rmse")
    predicted_df = dt_pipelineModel.transform(test_df)
    rmse = evaluator.evaluate(predicted_df)
    ##TrainValidationSplit训练找出最佳模型
    paramGrid = ParamGridBuilder().addGrid(dt.impurity, ["gini", "entory"]).addGrid(dt.maxDepth, [5, 10, 15]).addGrid(
        dt.maxBins, [10, 15, 20]).build()
    cv=CrossValidator(estimator=dt, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=3)
    cv_pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])
    cv_pipelineModel = cv_pipeline.fit(dataset=train_df)
    ##使用最佳模型进行预测
    predictions =cv_pipelineModel.transform(test_df)
    rmse2 = evaluator.evaluate(predictions)
    print(rmse2)

def func3():
    """
    使用GBT ,梯度提升树
    :return:
    """
    hour_df = sqlContext.read.format("csv").option("header", "true").load(Path + "hour.csv")
    # 舍弃不需要的字段
    hour_df = hour_df.drop("instant").drop("dteday").drop("yr").drop("casual").drop("registered")
    # 数据转换为double
    hour_df = hour_df.select([col(column).cast("double").alias(column) for column in hour_df.columns])
    # 将数据分为train_df和test_df，比例为0.7:0.3
    train_df, test_df = hour_df.randomSplit([0.7, 0.3])
    train_df.cache()
    test_df.cache()
    # 创建特征字段list
    featureCols = hour_df.columns[:-1]
    ###评估模型
    evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="rmse")
    # 建立pipeline
    vectorAssembler = VectorAssembler(inputCols=featureCols, outputCol="aFeatures")
    vectorIndexer = VectorIndexer(inputCol="aFeatures", outputCol="features", maxCategories=24)
    gbt=GBTRegressor(labelCol="cnt",featuresCol="features")
    gbt_pipeline=Pipeline(stages=[vectorAssembler,vectorIndexer,gbt])
    gbt_pipelineModel=gbt_pipeline.fit(train_df)
    predicted_df=gbt_pipelineModel.transform(test_df)
    rmse=evaluator.evaluate(predicted_df)
    print(rmse)

    ##交叉验证
    paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [5,10]).addGrid( gbt.maxBins, [25,40]).addGrid(gbt.maxIter, [10, 50]).build()
    cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=3)
    cv_pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])
    cv_pipelineModel = cv_pipeline.fit(dataset=train_df)
    ##使用最佳模型进行预测
    predictions = cv_pipelineModel.transform(test_df)
    rmse2 = evaluator.evaluate(predictions)
    print(rmse2)




























