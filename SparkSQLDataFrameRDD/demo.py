from pyspark import SparkContext
from pyspark.sql import Row, SparkSession
import pandas as pd
import matplotlib.pyplot as plt

"""
Spark sql ,dataFrame RDD使用
"""
sc = SparkContext('local')
Path = "D:\\workingSpace\\pythonAndSpark2\data\\ml-100k\\"
sqlContext = SparkSession.builder.getOrCreate()


def func1():
    rawUserRdd = sc.textFile(Path + "u.user")
    print("数据量 rawUserRdd.count():=", rawUserRdd.count())
    print("查看前2行：", rawUserRdd.take(2))
    user_Rows = rawUserRdd.map(lambda p: Row(
        userid=int(p[0]),
        age=int(p[1]),
        gender=p[2],
        occupation=p[3],
        zipcode=p[4]
    ))
    print("dataFrame 的前3，user_Rows.take(3):", user_Rows.take(3))
    user_df = sqlContext.createDataFrame(user_Rows)
    # 展示schema，类似表结构
    print("#展示schema，类似表结构:")
    user_df.printSchema()
    # 展示前3个数据
    print("#展示前3个数据:")
    user_df.show(3)
    # dataFrame创建别名
    df = user_df.alias("df")
    print("#dataFrame创建别名:")
    df.show(3)
    df.registerTempTable("user_table")
    print("sparkSQL 查询条数:")
    sqlContext.sql("select count(*) counts from user_table").show()
    # 多行输入，3引号的使用
    print("sparkSQL 查询条数2:")
    sqlContext.sql("""select count(*) counts 
    from user_table""").show()
    print("sparkSQL 查询数据(默认前20条):")
    sqlContext.sql("select *  from user_table").show()
    print("sparkSQL 查询数据(指定3条):")
    sqlContext.sql("select *  from user_table").show(3)
    print("sparkSQL 查询数据(指定3条使用limit，可减少运行时间):")
    sqlContext.sql("select *  from user_table limit 3").show()

    ######选择指定字段展示的三种方式，RDD,dataFrame,sql
    # RDD
    userRDDnew = rawUserRdd.map(lambda x: (x[0], x[3], x[2], x[1]))  # 选取字段
    print("使用RDD方式选取字段展示：", userRDDnew.take(3))
    # 使用dataFrame选取字段
    print("#使用dataFrame选取字段,输入字段名称字符串：")
    user_df.select("userid", "occupation", "gender", "age").show(3)
    print("#使用dataFrame选取字段,dataFrame.字段名，(dataFrame使用创建的别名也行如：df.userid，df.occupation，或者中括号也行df['occupation']选取字段：")
    user_df.select(user_df.userid, user_df.occupation, user_df.gender, user_df.age).show(3)
    # spark sql
    sqlContext.sql("select userid,occupation,gender,age  from user_table limit 3").show()

    #####增加计算字段，即有些字段数据需要计算得到
    # RDD
    userRDDnew2 = rawUserRdd.map(lambda x: (x[0], x[3], x[2], 2016 - int(x[1])))
    print("RDD计算字段：", userRDDnew2.take(3))
    # dataFrames计算值并娶一个别名，不然字段名就为2016-df.age
    print("dataframe计算字段：")
    df.select("userid", "occupation", "gender", (2016 - df.age).alias("birthyear")).show(3)
    # sparksql
    print("sparksql：")
    sqlContext.sql("select userid,occupation,gender,2016-age birthyear  from user_table").show(3)

    ######删选数据 类似where条件
    # RDD
    print("使用RDD筛选，lambda表达式：",
          rawUserRdd.filter(lambda r: r[3] == "technician" and r[2] == "M" and r[1] == 24).take(3))
    # dataframes
    # 1多个filter 相当于and
    user_df.filter("occupation='technician'").filter("gender='M'").filter("age=24").show()
    # 2单个filter配合and or not
    user_df.filter("occupation='technician'" and "gender='M'" and "age=24").show()
    # 3使用[名称].[字段] 方式，=要为==，and要为&,中括号引用类似
    df.filter((df.occupation == "technician") & (df.gerder == "M") & (df.age == 24)).show()
    # sparksql,很简单，类似sql添加where调价即可
    sqlContext.sql(
        "select userid,occupation,gender,age  from user_table where occupation='technician' and gender='M' and age=24").show(
        3)

    #####排序
    # RDD  takeOrdered
    print("RDD 排序默认升序：", rawUserRdd.takeOrdered(3, key=lambda x: int(x[1])))
    print("RDD 排序，降序（取反）：", rawUserRdd.takeOrdered(3, key=lambda x: -1 * int(x[1])))
    # dataframes
    # 1升序,默认升序
    user_df.select("userid", "occupation", "gender", "age").orderBy("age").show(3)
    user_df.select("userid", "occupation", "gender", "age").orderBy(df.age).show(3)
    # 2降序
    user_df.select("userid", "occupation", "gender", "age").orderBy("age", ascending=0).show(3)
    user_df.select("userid", "occupation", "gender", "age").orderBy(df.age.desc()).show(3)
    # sparksql order by desc,asc
    sqlContext.sql("select userid,occupation,gender,age  from user_table order by age asc").show(3)
    sqlContext.sql("select userid,occupation,gender,age  from user_table order by age desc").show(3)
    ####按照多个字段排序
    # rdd
    print("RDD 多字段排序：", rawUserRdd.takeOrdered(3, key=lambda x: (-int(x[1]), x[2])))  # 现x1降序再x2升序
    # dataframes
    df.orderBy(["age", "gender"], ascending=[0, 1]).show(3)  # 0表示升序1表示降序
    df.orderBy(df.age.desc(), df.gender).show(3)
    # sparksql
    sqlContext.sql("select userid,occupation,gender,age  from user_table order by age desc,gender asc").show(3)

    #####去重
    # rdd
    print("RDd 去重：", rawUserRdd.map(lambda x: x[2]).distinct().collect())
    # 限制多个字段，类似双主键
    print("RDD 去重多字段：", rawUserRdd.map(lambda x: (x[1], x[2])).distinct().take(5))
    # dataframes
    user_df.select("gender").distinct().show()
    user_df.select("age", "gender").distinct().show()  # 多字段
    # sparksql
    sqlContext.sql("select distinct gender from user_table ").show()

    ####分组，统计
    # rdd
    print("RDD分组统计：", rawUserRdd.map(lambda x: (x[2], 1)).reduceByKey(
        lambda x, y: x + y).collect())  # map将数据变成（性别，1），reduce分别按照性别统计和
    print("RDD分组统计，多字段：",
          rawUserRdd.map(lambda x: ((x[2], x[3]), 1)).reduceByKey(lambda x, y: x + y).collect())  # 按照性别职业来统计数据
    # dataframes
    user_df.select("gender").groupBy("gender").count().show()
    user_df.select("gender", "occupation").groupBy("gender", "occupation").count().orderBy("gender", "occupation").show(
        10)
    # TODO crosstad
    user_df.stat.crosstab("occupation", "gender").show(10)
    # sparksql
    sqlContext.sql("select  gender,count(*) counts from user_table group by gender").show()
    sqlContext.sql("select  gender,occupation,count(*) counts from user_table group by gender,occupation").show(10)

    ###创建邮编数据
    ZipCodeRDD = getZipcode()
    zipcode_data = ZipCodeRDD.map(lambda p: Row(
        zipcode=int(p[0]),
        zipCodeType=p[1],
        city=p[2],
        state=p[3]
    ))
    print("zipcode前3：",zipcode_data.take(3))
    zipcode_df=sqlContext.createDataFrame(zipcode_data)
    zipcode_df.printSchema()
    #创建临时登陆表
    zipcode_df.registerTempTable("zipcode_table")
    zipcode_df.show(3)

    #####join 联接数据
    #sparksql
    sqlContext.sql("select  u.*,z.city,z.state from user_table u left join zipcode_table z on u.zipcode=z.zipcode where z.state='NY'").show(10)#查看纽约用户数据
    sqlContext.sql( "select z.state, count(*) from user_table u left join zipcode_table z on u.zipcode=z.zipcode group by z.state ").show( 10)  # 查看纽约用户数据
    #dataframes
    joined_df=user_df.join(zipcode_df,user_df.zipcode==zipcode_df.zipcode,"left_outer")
    print("dataframes 联接后：")
    joined_df.printSchema()
    #安州分组
    groupByState_df=joined_df.groupBy("state").count()

    groupByState_pandas_df=groupByState_df.toPandas().set_index("state")
    #画个直方图,安州统计数据
    ax=groupByState_pandas_df["count"].plot(kind="bar",title="State",figsize=(12,6),legend=True,fontsize=12)
    plt.show()

    #按照不同职业统计人数，并以圆饼图展示
    Occupation_df=sqlContext.sql("select u.occupation,count(*) counts from user_table u group by occupation")
    Occupation_pandas_df=Occupation_df.toPandas().set_index("occupation")
    ax2=Occupation_pandas_df["counts"].plot(kind="pie",title="occupation",figsize=(8,8),startangle=90,autopct="%1.1f%%")
    ax2.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
    plt.show()





def getZipcode():
    basepath = "D:\\workingSpace\\pythonAndSpark2\data\\zipcode\\"
    zipcode = sc.textFile(basepath + "zipcode.csv", minPartitions=20)
    header = zipcode.first()
    rawData = zipcode.filter(lambda x: x != header)
    data = rawData.map(lambda x: x.replace("\"", ""))
    ZipCodeRDD = data.map(lambda x: x.split(","))
    return ZipCodeRDD
