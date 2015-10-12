


from pyspark import SparkContext
from pyspark import SparkConf


def count_line_mapReduce():
  '''
  This function will count the number of lines in the file.
  It is implemented with mapReduce heuristics.
  '''
  # spark configuration
  APP_NAME = 'count lines in mapReduce'
  conf = SparkConf().setAppName(APP_NAME)
  conf = conf.setMaster('spark://ukko160:7077')
  sc = SparkContext(conf=conf)

  # mapReduce function call
  lines = sc.textFile('../spark-1.4.1-bin-hadoop2.6/README.md')
  lineLength = lines.map(count_line_map)
  totalLength = lineLength.reduce(count_line_reduce)
  return totalLength
  pass

def count_line_map(line):
  return len(line)
  pass

def count_line_reduce(length1, length2):
  return length1+length2
  pass


if __name__ == '__main__':
  print count_line_mapReduce()
