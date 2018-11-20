# -*- coding: utf-8 -*-
from pyspark import SparkContext
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.recommendation import ALS



sc = SparkContext('local')
global  Path
if sc.master[0:5] == "local":
    Path = "D:\\workingSpace\\pythonAndSpark2\data\\ml-100k\\"
rawUserData=sc.textFile(Path+"u.data")

#TODO简单测试
def config():
    print("总量：",rawUserData.count())
    print("查看第一项数据:",rawUserData.first())
    # 一行一条数据，每行按照制表符Tab 分割，取前三个字段
    #TODO map运算 传入一个函数，以rawUserData里面的数据为源数据，进行处理
    rawRatings=rawUserData.map(lambda line:line.split("\t")[:3])
    # 取五条数据
    print("取前五：",rawRatings.take(5))
    ratingsRDD=rawRatings.map(lambda x:(x[0],x[1],x[2]))
    print("ratingsRDD:",ratingsRDD.take(5))
    print("ratingsRDD项数：",ratingsRDD.count())
    print("查看不重复的用户数：",ratingsRDD.map(lambda x:x[0]).distinct().count())
    print("查看不重复的电影书：",ratingsRDD.map(lambda x:x[1]).distinct().count())

#TODO 训练数据,我们使用rawUserData 数据以map转换为rawRatings,再用map转换为ALS格式RDD[Rating],然后使用als.train训练
def trainData():
    """
    显示评分训练：
        def train(cls, ratings, rank, iterations=5, lambda_=0.01, blocks=-1, nonnegative=False, seed=None)
        rating：训练的数据格式，本例中为Rating（userID,productID,rating）的RDD
        rank:是指当我们矩阵分解Matrix Factorization时，将原本的矩阵A(m*n)分解成X(m*rank)[用户矩阵]与Y(rank*n)【产品矩阵】
        iterations:Als算法重复计算次数
        lambda_:
    隐世评分训练：
        trainImplicit
    :return:
    """
    rawRatings = rawUserData.map(lambda line: line.split("\t")[:3])
    ratingsRDD = rawRatings.map(lambda x: (x[0], x[1], x[2]))
    model=ALS.train(ratings=ratingsRDD,rank=10,iterations=10,lambda_=0.01)
    print(model)
    return model

#TODO 针对用户推荐电影，给用户推荐他可能感兴趣的电影
def recommendToUser():
    """
    使用model.recommendProducts(user:int,num:int)方法来推荐,输入user,针对这个user推荐num个感兴趣商品
    :return: 返回Rating数据类型（user:int,product:int,rating:double），user用户，product商品，rating推荐评分，rating越高表示越推荐
    """
    model=trainData()
    print("推荐用户前5的商品：",model.recommendProducts(user=100,num=5))
    print("查看针对用户推荐商品的评分：",model.predict(user=100,product=1141))

#TODO 针对电影推荐用户，如当我们想要促销某些电影时，将该电影推荐给可能感兴趣的用户
def recommendFilms():
    """
    model.recommendUsers(product:int,num:int),product要被推荐的商品编号，num：需要推荐的用户数
    :return: 返回Rating数据类型（user:int,product:int,rating:double），user用户，product商品，rating推荐评分，rating越高表示越推荐
    """
    model=trainData()
    print("针对电影200推荐前5个用户：",model.recommendUsers(product=200,num=5))

def mapTheNameAndId():
    """
    u.item里面是电影数据，查看文件中的数据可发现，数据格式为：编号|名称|。。。。，分别以|分割，取出前两项就为编号，电影名称映射
    :return:
    """
    itemRDD=sc.textFile(Path+"u.item")
    print("电影数：",itemRDD.count())
    #先分割，在去取前两项，在组成map
    movieTitle=itemRDD.map(lambda line:line.split("|")).map(lambda x:(float(x[0]),x[1])).collectAsMap()
    print("map的长度：",len(movieTitle))
    #注意要转换为list，不然报错
    print("显示前5项数据：",list(movieTitle.items())[:5])
    print("查询电影名称，如5",movieTitle[5])
    print("显示前5条电影名称:")
    model = trainData()
    recommendP=model.recommendProducts(user=100,num=5)
    for p in recommendP:
        print("对用户",str(p[0]) ,"推荐电影",str(movieTitle[p[1]]) ,"推荐评分",p[2])

if __name__=="__main__":
    # recommendToUser()
    recommendFilms()
    # mapTheNameAndId()