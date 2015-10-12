
from pyspark.streaming import StreamingContext
from pyspark import SparkContext
from pyspark import SparkConf


def words_count_mapReduce():
   
  # configuration
  APP_NAME = 'word count'
  conf = SparkConf().setAppName(APP_NAME)
  conf = conf.setMaster('spark://ukko160:7077')
  sc = SparkContext(conf=conf)

  # actual function
  lines = sc.textFile("../spark-1.4.1-bin-hadoop2.6/README.md")
  table = lines.flatMap(f_flatmapper).map(f_mapper).reduce(f_reducer)
  for x in table:
    print x

  pass

def f_flatmapper(line):
  return line.strip().split(' ')

def f_mapper(word):
  return (word,1)

def f_reducer(a,b):
  return a+b

if __name__ == '__main__':
  words_count_mapReduce()

