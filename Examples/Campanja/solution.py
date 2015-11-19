
# python packages included in this scsript
from pyspark import SparkContext
from pyspark import SparkConf

# meta information
__author__ = 'Hongyu Su'
__version__ = '0.1'

def statistics():
  '''
  this function is designed to 
  '''
  # configuration
  APP_NAME = 'statistics'
  conf = SparkConf().setAppName(APP_NAME)
  #conf = conf.setMaster('spark://ukko160:7077')
  conf = conf.setMaster('local[2]')
  sc = SparkContext(conf=conf)

  # actuall lambda
  lines = sc.textFile('../spark-1.4.1-bin-hadoop2.6/README.md')
  lineLength = lines.map(lambda s: len(s))
  totalLength = lineLength.reduce(lambda a,b: a+b)
  return totalLength
  pass



if __name__ == '__main__':
  statistics()