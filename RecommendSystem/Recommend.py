from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
import RecommendSystem.RecommendTrain as train

Path = "D:\\workingSpace\\pythonAndSpark2\data\\ml-100k\\"


def CreateSparkContext():
    sc = SparkContext('local')
    return sc


def PrepareData(sc):
    """
    u.item里面是电影数据，查看文件中的数据可发现，数据格式为：编号|名称|。。。。，分别以|分割，取出前两项就为编号，电影名称映射
    :return:
    """
    itemRDD = sc.textFile(Path + "u.item")
    # 先分割，在去取前两项，在组成map
    movieTitle = itemRDD.map(lambda line: line.split("|")).map(lambda x: (float(x[0]), x[1])).collectAsMap()
    return movieTitle


def Loadmodel(sc):
    """
    windows环境下会报错，file permision什么的，现拿现跑
    :param sc:
    :return:
    """
    try:
        path = "D:\\workingSpace\\pythonAndSpark2\\RecommendSystem\\model\\ALSmodel\\"
        print(path)
        model = MatrixFactorizationModel.load(sc, path)
        print("载入模型成功")
    except Exception:
        print("载入模型失败，可能路径不存在")
        model=train.getModel(sc)
    return model


# 给用户推荐电影
def RecommendMovies(model, movieTitle, userId):
    recommendMovies = model.recommendProducts(userId, 10)
    for p in recommendMovies:
        print("对用户", str(p[0]), "推荐电影", str(movieTitle[p[1]]), "推荐评分", p[2])


def RecommendUsers(model, movieTitle, movieId):
    recommendUsers = model.recommendUsers(movieId, 10)
    print("针对电影id:{0},电影名称：{1},推荐以下用户：".format(movieId, movieTitle[movieId]))
    for p in recommendUsers:
        #(user=818, product=200, rating=5.669371781181173)
        print("针对用户id:{0},推荐评分：{1}".format(p[0], p[2]))


def Recommend(type, id, model, movieTitle):
    if type == "U" or type == "u":
        recommend = RecommendMovies(model, movieTitle, id)
    if type == "M" or type == "m":
        recommend = RecommendUsers(model, movieTitle, id)


if __name__ == "__main__":
    type = input("please input recommend type(u for user,m for product):")
    id = input("please input recommend id:")
    id=int(id)
    sc = CreateSparkContext()
    print("==========数据准备=============")
    movieTitle = PrepareData(sc)
    print("==========载入模型=============")
    model = Loadmodel(sc)
    print("==========进行推荐==============")
    Recommend(type=type, id=id, model=model, movieTitle=movieTitle)
