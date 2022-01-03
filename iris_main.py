from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, DoubleType, StringType, IntegerType
import math
from operator import add
import numpy as np
import logging
from datetime import datetime

# config logging
logging.basicConfig(filename='app.log', filemode='w',
     format='%(asctime)s - %(levelname)s - %(message)s',
     level=logging.INFO)

# create spark session
spark = SparkSession.builder.appName("KMeansWithMapReduce").getOrCreate()

# data preprocessing
schema = StructType() \
    .add("sepal_length", DoubleType(), True) \
    .add("sepal_width", DoubleType(), True) \
    .add("petal_length", DoubleType(), True) \
    .add("petal_width", DoubleType(), True) \
    .add("species", StringType(), True)

# set start time
start_time = datetime.timestamp(datetime.now())

# read dataset
df = spark.read.format("csv").option("header", True).schema(schema).load("/tmp/IRIS.csv")
df.na.drop()
logging.info("PHASE 1: Read the dataset.")

# k clusters
k = 3
# threshold between new and old centroid
threshold = 0.6

# randomly select centroids
centroids = df.rdd.takeSample(False, k, seed=0)
centroids = [[c.sepal_length, c.sepal_width, c.petal_length, c.petal_width] for c in centroids]
# centroid structure: (index, [centroid features]) e.g., (1, [2.0, 3.0, 5.0, 0.2])
centroids = [(idx, centroid) for idx, centroid in enumerate(centroids)]
logging.info("PHASE 2: Randomly select centroids.")

# set points
points = df.rdd
# point structure: ([point features], count) e.g., ([2.0, 5.0, 2.5, 0.6], 1)
points_rdd = points.map(lambda p: ([p.sepal_length, p.sepal_width, p.petal_length, p.petal_width], 1))
points_rdd.cache()
logging.info("PHASE 3: Points convert into RDD.")

""" calculate distance """
def calculateDistance(point, centroid):
    distance = 0
    for index in range(len(point)):
        distance += (point[index]-centroid[index])**2
    return math.sqrt(distance)

""" belongs to Centroid """
def belongCluster(point, centroids):
    centroidIndex = 0
    closest = float("+inf")
    for centroid in centroids:
        dist = calculateDistance(point, centroid[1])
        if dist < closest:
            closest = dist
            centroidIndex = centroid[0]
    return centroidIndex

""" Reduce all points in each centroid """
def accumulatedCluster(p1, p2):
    cluster_sum = list(map(add, p1[0], p2[0]))
    cluster_count = p1[1]+p2[1]
    p = [cluster_sum, cluster_count]
    return p

# training
maxRound = 100
round = 0

logging.info('PHASE 4: Start training')
while(maxRound > round):
    round += 1
    logging.info("Round: " + str(round))
    
    # Map Phase
    pointMapCentroid_rdd = points_rdd.keyBy(lambda point: belongCluster(point[0], centroids))
    logging.info('PHASE 4-1: MAP')

    # Reduce Phase
    pointReducedCentroid_rdd = pointMapCentroid_rdd.reduceByKey(lambda p1, p2: accumulatedCluster(p1, p2))
    pointReducedCentroid_rdd = pointReducedCentroid_rdd.map(lambda p: (p[0], np.divide(p[1][0], p[1][1]).tolist()))
    reduced_points = pointReducedCentroid_rdd.collect()
    logging.info('PHASE 4-2: REDUCE')

    # create new centroids
    new_centroids = sorted(reduced_points)
    centroids.sort()

    # convergence
    convergence_percentage = 0
    for index, centroid in enumerate(centroids):
        dist = calculateDistance(centroid[1], new_centroids[index][1])
        
        if dist < threshold:
            convergence_percentage += 1
    logging.info('PHASE 4-3: Convergence')
            
    centroids = new_centroids
    percentage = len(centroids)*80/100

    if convergence_percentage > percentage:
        logging.info("PHASE 4-4: Centroids converged")
        break

# stop time
end_time = datetime.timestamp(datetime.now())

# write files
logging.info('PHASE 5: Write files')
schema_template1 = StructType() \
    .add("cluster_index", IntegerType(), True) \
    .add("sepal_length", DoubleType(), True) \
    .add("sepal_width", DoubleType(), True) \
    .add("petal_length", DoubleType(), True) \
    .add("petal_width", DoubleType(), True)

# Q1: what cluster does each point maps into? 
df_r1 = spark.createDataFrame(pointMapCentroid_rdd.map(lambda p: (p[0], p[1][0][0], p[1][0][1],
                                                                    p[1][0][2], p[1][0][3])), schema=schema_template1)
df_r1.coalesce(1).write.option("header", True).csv("/tmp/r1_iris_pointMapCluster")
logging.info('PHASE 5-1: write result of Q1')

# Q2: what are coordinate of centroid points?
df_r2 = spark.createDataFrame(spark.sparkContext.parallelize(centroids).map(lambda p: (p[0], p[1][0], p[1][1],
                                                    p[1][2], p[1][3])), schema=schema_template1)
df_r2.coalesce(1).write.option("header", True).csv("/tmp/r2_iris_eachCentroidCoordinate")
logging.info('PHASE 5-2: write result of Q2')

# Q3: what's cost time?
r3 = end_time - start_time
logging.info("PHASE 5-3: write result of Q3 and cost time- "+str(r3))

spark.stop()


