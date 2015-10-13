


from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
import itertools
from math import sqrt
import sys
from operator import add
from os.path import join, isfile, dirname

def parseRating(line):
  """
  Parses a rating record in MovieLens format userId::movieId::rating::timestamp.
  """
  fields = line.strip().split("::")
  print long(fields[0]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))
  return long(fields[0]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))




def cf():
  '''
  collaborative filtering approach for movie recommendation
  file format UserID::MovieID::Rating::Time
  '''
  # set up Spark environment
  APP_NAME = "Collaboratove filtering for movie recommendation"
  conf = SparkConf().setAppName(APP_NAME)
  conf = conf.setMaster('spark://ukko160:7077')
  sc = SparkContext(conf=conf)

  data = sc.textFile('../Data/ml-1m/ratings.dat')
  ratings = data.map(parseRating)
  numRatings  = ratings.count()
  numUsers    = ratings.values().map(lambda r:r[0]).distinct().count()
  numMovies   = ratings.values().map(lambda r:r[1]).distinct().count()
  print "--- %d ratings from %d users for %d movies\n" % (numRatings, numUsers, numMovies)

  numPartitions = 10
  training    = ratings.filter(lambda r:r).values().repartition(numPartitions).cache()
  validation  = ratings.filter(lambda r:r).values().repartition(numPartitions).cache()
  test        = ratings.filter(lambda r:r).values().cache()
  numTraining         = training.count()
  numValidation       = validation.count()
  numTest             = test.count()
  print '-----',ratings.count(), training.count(),validation.count(),test.count()

  # training with ALS
  ranks       = [8,12]
  lambdas     = [0.1,0.01]
  numIters    = [10,20]
  bestModel   = None
  bestValidationRmse = float("inf")
  bestRank    = 0
  bestLambda  = -1.0
  bestNumIter = -1
  for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
    model                   = ALS.train(training, rank, numIter, lmbda)
    predictions             = model.predictAll(validation.map(lambda x:(x[0],x[1])))
    predictionsAndRatings   = predictions.map(lambda x:((x[0],x[1]),x[2])).join(validation.map(lambda x:((x[0],x[1]),x[2]))).values()
    validationRmse          = sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(numIter))
    print rank, lmbda, numIter, validationRmse
    
    if (validationRmse < bestValidationRmse):
      bestModel = model
      bestValidationRmse = validationRmse
      bestRank = rank
      bestLambda = lmbda
      bestNumIter = numIter
  print bestRank, bestLambda, bestNumIter, bestValidationRmse 

  sc.stop()



if __name__ == '__main__':
  cf()


