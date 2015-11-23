
import numpy as np
import logging
from pyspark.mllib.recommendation import ALS
from math import sqrt
import itertools
from operator import add
import plotting
import math

from sklearn.ensemble import RandomForestRegressor


logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.WARNING)



def mapper_als(line):
  '''
  mapper function for missing_value_imputation
  '''
  r = []
  res = {}
  for date,val in line[1]:
    if not date in res:
      res[date] = [val,1]
    else:
      res[date][0]+=val
      res[date][1]+=1
  newres = []
  for date in range(93):
    if date in res.keys():
      newres.append( (date,res[date][0]/float(res[date][1])) )
    else:
      newres.append( (date,None) )
  return (line[0],newres)
  pass

def missing_value_imputation(lines,index):
  '''
  collaborative filtering for missing value imputation
  '''
  logging.warning('--------------------------------- imputation on index %d -----------------------------',index)
  #--------------------------------- collect data from spark ---------------------------------------
  data = lines.map( lambda x: ( x[1], [(x[2],x[index])]  ) ) \
              .reduceByKey( lambda x,y: x+y ) \
              .map(mapper_als) \
              .flatMap(lambda x: [(x[0],date,val) for date,val in x[1]] ) \
              .map(lambda x: ( (x[0],x[1]), (x) ) )

  logging.warning("Entry: (missing) %d (all) %d (missing at) %.2f %% (keywords) %d" % \
                  (data.filter(lambda x: x[1][2]==None).count(), \
                   data.count(), \
                   100*data.filter(lambda x: x[1][2]==None).count() / float(data.count()), \
                   data.map(lambda x: x[0][0]).distinct().count() ))

  #--------------------------------- missing value imputation with collaborative filtering ALS ---------------------
  training = data.filter(lambda x: not x[1][2]==None).values().cache()
  test     = data.filter(lambda x:     x[1][2]==None).values().cache()
  numTraining = training.count()
  numTest     = test.count()
  logging.warning('Number of training:\t %d (keyword) %d' % (numTraining,training.map(lambda x: x[0]).distinct().count()))
  logging.warning('Number of test:    \t %d (keyword) %d' % (numTest,        test.map(lambda x: x[0]).distinct().count()))

  # model training with parameter selection on the validation dataset
  ranks       = [30,20,10]
  lambdas     = [0.1,0.01,0.001]
  numIters    = [10,15,20]
  #ranks = [10]
  #lambdas = [0.1]
  #numIters = [2]
  bestModel   = None
  bestValidationRmse = float("inf")
  bestRank    = 0
  bestLambda  = -1.0
  bestNumIter = -1
  for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
    model                    = ALS.train(training, rank, numIter, lmbda)
    predictions              = model.predictAll(training.map(lambda x:(x[0],x[1])))
    predictionsAndTrueValues = predictions.map(lambda x:((x[0],x[1]),x[2])).join(training.map(lambda x:((x[0],x[1]),x[2]))).values()
    validationRmse           = sqrt(predictionsAndTrueValues.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(numTraining))
    logging.warning("(rank) %d\t(lmbda) %.3f\t(numIter) %d\t(validationRmse) %.2f"   %(rank, lmbda, numIter, validationRmse))
    if (validationRmse < bestValidationRmse):
      bestValidationRmse = validationRmse
      bestRank = rank
      bestLambda = lmbda
      bestNumIter = numIter
  logging.warning("(rank) %d\t(lmbda) %.3f\t(numIter) %d\t(validationRmse) %.2f (best parameters)" \
                   %(bestRank, bestLambda, bestNumIter, bestValidationRmse))
  logging.warning("ALS  imputation RMSE:\t%.2f" % bestValidationRmse)
  # best ALS model
  bestModel = ALS.train(training, bestRank, bestNumIter, bestLambda)
 
  # base line imputation model
  meanRating = training.map(lambda x: x[2]).mean()
  baselineRmse = sqrt(training.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / numTraining)
  logging.warning("Mean imputation RMSE:\t%.2f" % baselineRmse)

  # predict test value
  try:
    predictions              = bestModel.predictAll(test.map(lambda x:(x[0],x[1])))
    predictionsAndTrueValues = predictions.map(lambda x:((x[0],x[1]),x[2])).join(test.map(lambda x:((x[0],x[1]),x[2])))
  except Exception as myerror:
    print myerror
  
  training    = training.map(lambda x:    (x[0],[(x[1],x[2],1)]) )
  predictions = predictions.map(lambda x: (x[0],[(x[1],x[2],0)]) )
  results     = training.union(predictions).reduceByKey(lambda x,y: x+y)

  return results
  pass


def mapper_regression(line):
  '''
  mapper function for local regression
  the mapper function works on individual keywork
  a randomforest regression model is applied to predict the next time point
  '''
  npdata = np.array(line[1])
  npdata = npdata[npdata[:,0].argsort()]
  f_tr = []
  l_tr = []
  for i in range(2,86):
    f_tr.append(npdata[i:(i+7),1])
    l_tr.append(npdata[i+6,1])
  f_ts = npdata[86:93,1]
  clf = RandomForestRegressor(n_estimators=150, min_samples_split=1)
  clf.fit(f_tr, l_tr)
  y_tr = clf.predict(f_tr)
  # return (keyword,(predicted value, training RMSE))
  return ( line[0], (clf.predict(f_ts).tolist()[0], math.sqrt(np.sum(clf.predict(f_tr)-l_tr)**2)) )
  pass

def local_regression(data,name):
  '''
  regression model wrapper
  '''
  logging.warning('--------------------------------- regression: %s --------------------------------- ' % name)
  #data = data.filter(lambda x: x[0]<10).map(mapper_regression)
  data = data.map(mapper_regression)
  return data
  pass





