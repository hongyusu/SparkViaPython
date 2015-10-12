
from pyspark.streaming import StreamingContext
from pyspark import SparkContext
from pyspark import SparkConf


def words_count_lambdaExpression():
   
  # configuration
  APP_NAME = 'words count with python lambda expression'
  conf = SparkConf().setAppName(APP_NAME)
  conf = conf.setMaster('spark://ukko160:7077')
  sc = SparkContext(conf=conf)

  # actual function
  lines = sc.textFile("../spark-1.4.1-bin-hadoop2.6/README.md")
  words = lines.flatMap(lambda x: x.split(' '))
  pairs = words.map(lambda x: (x,1))
  count = pairs.reduceByKey(lambda x,y: x+y)

  for x in count.collect():
    print x

  pass

if __name__ == '__main__':
  words_count_lambdaExpression()

