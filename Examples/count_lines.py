

from pyspark import SparkContext
from pyspark import SparkConf

def count_lines_lambdaexpression():
  
  # configuration
  APP_NAME = 'count lines'
  conf = SparkConf().setAppName(APP_NAME)
  conf = conf.setMaster('spark://ukko160:7077')
  sc = SparkContext(conf=conf)

  # core part of the script
  lines = sc.textFile("../spark-1.4.1-bin-hadoop2.6/README.md")
  lineLength = lines.map(lambda s: len(s))
  totalLength = lineLength.reduce(lambda a,b:a+b)

  # output results
  print totalLength

def count_lines_functioncall():
  
  # configuration
  APP_NAME = 'count lines'
  conf = SparkConf().setAppName(APP_NAME)
  conf = conf.setMaster('spark://ukko160:7077')
  sc = SparkContext(conf=conf)

  # core part of the script
  lines = sc.textFile("../spark-1.4.1-bin-hadoop2.6/README.md")
  lineLength = lines.map(count_lines_single)
  totalLength = lineLength.reduce(reducer)

  # output results
  print totalLength

def count_lines_single(lines):
  return len(lines)

def reducer(length1,length2):
  return length1+length2
  


if __name__ == '__main__':
  #count_lines_lambdaexpression()
  count_lines_functioncall()
