
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
  conf = conf.setMaster('spark://ukko160:7077')
  sc = SparkContext(conf=conf)

  # actuall lambda
  lines = sc.textFile('./campanja_work_sample_data_customerA.csv')
  lineLength = lines.map(lambda s: 1 )
  totalLength = lineLength.reduce(lambda a,b: a+b)
  return totalLength
  pass



if __name__ == '__main__':
  totalLength = statistics()
  print '----',totalLength
