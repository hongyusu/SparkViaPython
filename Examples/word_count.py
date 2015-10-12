
from pyspark.streaming import StreamingContext
from pyspark import SparkContext
from pyspark import SparkConf

def word_count_functioncall():
  
  # configuration
  APP_NAME = 'word count'
  conf = SparkConf().setAppName(APP_NAME)
  conf = conf.setMaster('spark://ukko178:7077')
  sc = SparkContext(conf=conf)

  # core part of the script
  lines = sc.textFile("../spark-1.4.1-bin-hadoop2.6/README.md")
  table = lines.flatMap(flatmapper).map(mapper).reduceByKey(reducer)
  for x in table.collect():
    print x

def flatmapper(lines):
  return lines.split(' ')
  
def mapper(word):
  return (word,1)

def reducer(a, b):
  return a+b
 
def word_count_lambdaexpression():
  # configuration
  APP_NAME = 'word count'
  conf = SparkConf().setAppName(APP_NAME)
  conf = conf.setMaster('spark://ukko178:7077')
  sc = SparkContext(conf=conf)

  # core part of the script
  lines = sc.textFile("../spark-1.4.1-bin-hadoop2.6/README.md")
  words = lines.flatMap(lambda x: x.split(' '))
  pairs = words.map(lambda x: (x,1))
  count = pairs.reduceByKey(lambda x,y: x+y)

  # output results
  for x in count.collect():
    print x

if __name__ == '__main__':
  word_count_functioncall()
  #word_count_lambdaexpression()
