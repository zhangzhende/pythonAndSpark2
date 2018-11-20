from pyspark import SparkContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import StringType,StructType,StructField
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder,TrainValidationSplit,CrossValidator
"""
使用pipeline机器学习流程多远分类
"""
sc = SparkContext('local')
Path = "D:\\workingSpace\\pythonAndSpark2\data\\covtype\\"
sqlContext = SparkSession.builder.getOrCreate()

def func1():
    rawData=sc.textFile(Path+"covtype.data",minPartitions=40)
    lines=rawData.map(lambda x:x.split(","))
    print("lines.count()::",lines.count())
    fieldNum=len(lines.first())
    print("字段数：",fieldNum)
    fields=[StructField(name="f"+str(i),dataType=StringType,nullable=True) for i in range(fieldNum)]
    schema=StringType(fields)
    covtype_df=sqlContext.createDataFrame(data=lines,schema=schema)
    print("covtype_df.columns::",covtype_df.columns)
    print("covtype_df.printSchema()::",covtype_df.printSchema())
    #数据转换为double
    covtype_df=covtype_df.select([col(column).cast("double").alias(column) for column in covtype_df.columns])
    print("装换后：covtype_df.printSchema()：",covtype_df.printSchema())
    #创建特征字段list
    featureCols=covtype_df.columns[:54]
    print("featureCols:",featureCols)
    #设置label字段,第54个字段是label，值范围是1-7，但是训练需从0开始，所以covtype_df["f54"]-1，表示将值都范围转到0-6了
    covtype_df=covtype_df.withColumn(colName="label",col=covtype_df["f54"]-1).drop("f54")
    print("第一项数据：",covtype_df.show(1))
    #将数据分为train_df和test_df，比例为0.7:0.3
    train_df,test_df=covtype_df.randomSplit([0.7,0.3])
    train_df.cache()
    test_df.cache()
    #建立pipeline
    vectorAssembler = VectorAssembler(inputCols=featureCols, outputCol="features")
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=5, maxBins=20)
    dt_pipeline = Pipeline(stages=[vectorAssembler, dt])
    print("查看pipeline流程：",dt_pipeline.getStages())
    #训练
    pipelineModel=dt_pipeline.fit(dataset=train_df)
    print("查看训练完成后的模型：",pipelineModel.stages[1].toDebugString[:500])
    #使用transform预测
    predicted=pipelineModel.transform(test_df)
    print("查看新增的字段：",predicted.columns)
    print("查看预测的结果：",predicted.show(2))
    ###评估模型
    evaluator=MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="accuracy")
    accuracy=evaluator.evaluate(predicted)
    print("accuracy::",accuracy)
    ##TrainValidationSplit训练找出最佳模型
    paramGrid = ParamGridBuilder().addGrid(dt.impurity, ["gini", "entory"]).addGrid(dt.maxDepth, [5, 10, 15]).addGrid(
        dt.maxBins, [10, 15, 20]).build()
    tvs = TrainValidationSplit(estimator=dt, evaluator=evaluator, estimatorParamMaps=paramGrid,
                               trainRatio=0.8)  # trainRatio 数据会8:2的比例分为训练集，验证集
    tvs_pipeline = Pipeline(stages=[vectorAssembler, tvs])
    pipelineModel = tvs_pipeline.fit(dataset=train_df)
    bestmodel=pipelineModel.stages[1].bestModel
    print("bestModel:",bestmodel.toDebugString[:500])
    ##使用最佳模型进行预测
    predictions=tvs_pipeline.transform(test_df)
    result=predictions.withColumnRenamed("f0","海拔").withColumnRenamed("f1", "方位").withColumnRenamed("f2","斜率")\
        .withColumnRenamed("f3","垂直距离").withColumnRenamed("f4","水平距离").withColumnRenamed("f5","阴影")
    result.select("海拔","方位","斜率","垂直距离","水平距离","阴影","label","prediction").show(10)
    accuracy2=evaluator.evaluate(predictions)
    print("accuracy2:",accuracy2)

def func2():
    rawData = sc.textFile(Path + "covtype.data", minPartitions=40)
    lines = rawData.map(lambda x: x.split(","))
    fieldNum = len(lines.first())
    print("字段数：", fieldNum)
    fields = [StructField(name="f" + str(i), dataType=StringType, nullable=True) for i in range(fieldNum)]
    schema = StringType(fields)
    covtype_df = sqlContext.createDataFrame(data=lines, schema=schema)
    covtype_df = covtype_df.select([col(column).cast("double").alias(column) for column in covtype_df.columns])
    # 创建特征字段list
    featureCols = covtype_df.columns[:54]
    vectorAssembler = VectorAssembler(inputCols=featureCols, outputCol="features")
    dt = DecisionTreeClassifier(labelCol="label", featuresCol="features",  maxDepth=5, maxBins=20)
    dt_pipeline=Pipeline(stages=[vectorAssembler,dt])



























