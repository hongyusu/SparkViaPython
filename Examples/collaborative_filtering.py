


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
  return (int(int(fields[0])%10),int(int(fields[1])%10)), (int(fields[0]), int(fields[1]), float(fields[2]))

def cf(filename):
  '''
  collaborative filtering approach for movie recommendation
  file format UserID::MovieID::Rating::Time
  '''
  # set up Spark environment
  APP_NAME = "Collaboratove filtering for movie recommendation"
  conf = SparkConf().setAppName(APP_NAME)
  conf = conf.setMaster('spark://ukko160:7077')
  sc = SparkContext(conf=conf)

  # read in data
  data = sc.textFile(filename)
  ratings = data.map(parseRating)
  numRatings  = ratings.count()
  numUsers    = ratings.values().map(lambda r:r[0]).distinct().count()
  numMovies   = ratings.values().map(lambda r:r[1]).distinct().count()
  print "--- %d ratings from %d users for %d movies\n" % (numRatings, numUsers, numMovies)

  # select training and testing
  numPartitions = 10
  training    = ratings.filter(lambda r: not(r[0][0]<=0 and r[0][1]<=1) ).values().repartition(numPartitions).cache()
  test        = ratings.filter(lambda r: r[0][0]<=0 and r[0][1]<=1 ).values().cache()
  numTraining = training.count()
  numTest     = test.count()
  print "ratings:\t%d\ntraining:\t%d\ntest:\t\t%d\n" % (ratings.count(), training.count(),test.count())

  # model training with parameter selection on the validation dataset
  ranks       = [10,20,30]
  lambdas     = [0.1,0.01,0.001]
  numIters    = [10,20]
  bestModel   = None
  bestValidationRmse = float("inf")
  bestRank    = 0
  bestLambda  = -1.0
  bestNumIter = -1
  for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
    model                   = ALS.train(training, rank, numIter, lmbda)
    predictions             = model.predictAll(training.map(lambda x:(x[0],x[1])))
    predictionsAndRatings   = predictions.map(lambda x:((x[0],x[1]),x[2])).join(training.map(lambda x:((x[0],x[1]),x[2]))).values()
    validationRmse          = sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(numTraining))
    print rank, lmbda, numIter, validationRmse
    if (validationRmse < bestValidationRmse):
      bestModel = model
      bestValidationRmse = validationRmse
      bestRank = rank
      bestLambda = lmbda
      bestNumIter = numIter
  print bestRank, bestLambda, bestNumIter, bestValidationRmse 
  print "ALS on train:\t\t%.2f" % bestValidationRmse
  
  meanRating = training.map(lambda x: x[2]).mean()
  baselineRmse = sqrt(training.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / numTraining)
  print "Mean imputation:\t\t%.2f" % baselineRmse


  # predict test ratings
  try:
    predictions             = bestModel.predictAll(test.map(lambda x:(x[0],x[1])))
    predictionsAndRatings   = predictions.map(lambda x:((x[0],x[1]),x[2])).join(test.map(lambda x:((x[0],x[1]),x[2]))).values()
    testRmse          = sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(numTest))
  except Exception as myerror:
    print myerror
    testRmse          = sqrt(test.map(lambda x: (x[0] - 0) ** 2).reduce(add) / float(numTest))
  print "ALS on test:\t%.2f" % testRmse

  # use mean rating as predictions 
  meanRating = training.map(lambda x: x[2]).mean()
  baselineRmse = sqrt(test.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / numTest)
  print "Mean imputation:\t%.2f" % baselineRmse

  sc.stop()



if __name__ == '__main__':
  filenames = ['../Data/ml-1m/ratings.dat','../Data/ml-10M100K/ratings.dat']
  filenames = ['../Data/ml-10M100K/ratings.dat']
  for filename in filenames:
    cf(filename)


