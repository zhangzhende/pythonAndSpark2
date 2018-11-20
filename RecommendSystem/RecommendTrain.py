from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS

"""
相关注释在DEMO/demo1
"""
Path = "D:\\workingSpace\\pythonAndSpark2\data\\ml-100k\\"
RANK=5
ITERATION=10
LAMBDA_=0.1
def CreateSparkContext():
    sc = SparkContext('local')
    return sc
def PrepareData(sc):
    """
    训练数据,我们使用rawUserData 数据以map转换为rawRatings,再用map转换为ALS格式RDD[Rating],然后使用als.train训练
    训练的数据格式，本例中为Rating（userID,productID,rating）的RDD
    :param sc:
    :return:
    """
    rawUserData = sc.textFile(Path + "u.data")
    rawRatings = rawUserData.map(lambda line: line.split("\t")[:3])
    ratingsRDD = rawRatings.map(lambda x: (x[0], x[1], x[2]))
    return ratingsRDD
def SaveModel(model,sc):
    try:
        model.save(sc=sc, path="./model/ALSmodel")
        print("已存储在ALSmodel")
    except Exception:
        print("Model 已存在")

def getModel(sc):
    ratingsRDD = PrepareData(sc)
    model = ALS.train(ratings=ratingsRDD, rank=RANK, iterations=ITERATION, lambda_=LAMBDA_)
    return model
if __name__=="__main__":
  sc=CreateSparkContext()
  print("===========数据准备阶段====================")
  ratingsRDD=PrepareData(sc)
  print("===========数据训练阶段====================")
  print("开始ALS训练，参数，rank=",RANK,",iteratios=",ITERATION,",lambda_=",LAMBDA_)
  model=ALS.train(ratings=ratingsRDD,rank=RANK,iterations=ITERATION,lambda_=LAMBDA_)
  print("=============存储model====================")
  SaveModel(model=model,sc=sc)



