

from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionModel
from pyspark.mllib.regression import  LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.tree import DecisionTree,DecisionTreeModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
import itertools
import math
import random

  
def leastSquare(trainingData,testData,trainingSize,testSize):
  '''
  linear lr classifier
  '''
  # train a lr model
  numIterValList = [1000,3000,5000]
  stepSizeValList = [1e-11,1e-9,1e-7,1e-5]

  # variable for the best parameters
  bestNumIterVal = 200
  bestStepSizeVal = 1
  bestTrainingRMSE = 1e10 

  regParamVal = 0.0
  regTypeVal = None

  for numIterVal,stepSizeVal in itertools.product(numIterValList,stepSizeValList):
    model = LinearRegressionWithSGD.train(trainingData, iterations=numIterVal, step=stepSizeVal, regParam=regParamVal, regType=regTypeVal)
    ValsAndPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
    trainingRMSE = math.sqrt(ValsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / trainingSize)
    if trainingRMSE:
      if trainingRMSE<bestTrainingRMSE:
        bestNumIterVal = numIterVal
        bestStepSizeVal = stepSizeVal
        bestTrainingRMSE = trainingRMSE
    print numIterVal,stepSizeVal,trainingRMSE
  print bestNumIterVal,bestStepSizeVal,bestTrainingRMSE

  model = LinearRegressionWithSGD.train(trainingData, iterations=bestNumIterVal, step=bestStepSizeVal, regParam=regParamVal, regType=regTypeVal)

  # Evaluating the model on training data
  ValsAndPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
  trainingRMSE = math.sqrt(ValsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / trainingSize)
  print trainingRMSE

  # Evaluating the model on training data
  ValsAndPreds = testData.map(lambda p: (p.label, model.predict(p.features)))
  testRMSE = math.sqrt(ValsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / testSize)
  print testRMSE
  pass

def regularized(trainingData,testData,trainingSize,testSize,regTypeVal):
  '''
  Least square with l1 norm: lasso
  '''
  # train a lr model
  numIterValList = [3000,5000,10000]
  stepSizeValList = [1e-11,1e-9,1e-7]
  regParamValList = [0.01,0.1,1,10]

  # variable for the best parameters
  bestNumIterVal = 200
  bestStepSizeVal = 1
  bestTrainingRMSE = 1e10 
  bestRegParamVal = 0.0

  for numIterVal,stepSizeVal,regParamVal in itertools.product(numIterValList,stepSizeValList,regParamValList):
    model = LinearRegressionWithSGD.train(trainingData, iterations=numIterVal, step=stepSizeVal, regParam=regParamVal, regType=regTypeVal)
    ValsAndPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
    trainingRMSE = math.sqrt(ValsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / trainingSize)
    if trainingRMSE:
      if trainingRMSE<bestTrainingRMSE:
        bestNumIterVal = numIterVal
        bestStepSizeVal = stepSizeVal
        bestTrainingRMSE = trainingRMSE
    print numIterVal,stepSizeVal,regParamVal,trainingRMSE
  print bestNumIterVal,bestStepSizeVal,bestRegParamVal,bestTrainingRMSE

  model = LinearRegressionWithSGD.train(trainingData, iterations=bestNumIterVal, step=bestStepSizeVal, regParam=regParamVal, regType=regTypeVal)

  # Evaluating the model on training data
  ValsAndPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
  trainingRMSE = math.sqrt(ValsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / trainingSize)
  print trainingRMSE

  # Evaluating the model on training data
  ValsAndPreds = testData.map(lambda p: (p.label, model.predict(p.features)))
  testRMSE = math.sqrt(ValsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / testSize)
  print testRMSE
  pass


def decisionTreeRegression(trainingData,testData,trainingSize,testSize):
  '''
  decision tree for regression
  '''
  # parameter range
  maxDepthValList = [5,10,15]
  maxBinsVal = [16,32]

  # best parameters
  bestMaxDepthVal = 5
  bestMaxBinsVal = 16
  bestTrainingRMSE = 1e10

  for maxDepthVal,maxBinsVal in itertools.product(maxDepthValList,maxBinsVal):
    model = DecisionTree.trainRegressor(trainingData,categoricalFeaturesInfo={},impurity='variance',maxDepth=maxDepthVal,maxBins=maxBinsVal)
    predictions = model.predict(trainingData.map(lambda x:x.features))
    ValsAndPreds = trainingData.map(lambda x:x.label).zip(predictions)
    trainingRMSE = math.sqrt(ValsAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / trainingSize)
    if trainingRMSE:
      if trainingRMSE<bestTrainingRMSE:
        bestMaxDepthVal = maxDepthVal
        bestMaxBinsVal = maxBinsVal
        bestTrainingRMSE = trainingRMSE
    print maxDepthVal, maxBinsVal, trainingRMSE
    break
  print bestMaxDepthVal,bestMaxBinsVal,bestTrainingRMSE

  pass

if __name__ == '__main__':

  random.seed(1)

  # set up Spark environment
  APP_NAME = "Spark linear regression models"
  conf = SparkConf().setAppName(APP_NAME)
  conf = conf.setMaster('spark://ukko160:7077')
  sc = SparkContext(conf=conf)
  
  # load data from file
  parsedData = MLUtils.loadLibSVMFile(sc, "../Data/cadata")
  parsedData = MLUtils.loadLibSVMFile(sc, "../Data/YearPredictionMSD")

  # split data into training and test
  trainingData,testData = parsedData.randomSplit([0.8,0.2])
  trainingSize = trainingData.count()
  testSize = testData.count()
  print "Training:\t%d\nTest:\t\t%d" % (trainingSize,testSize)

  #trainingExamples = trainingData.collect()
  #testExamples = testData.collect()
  #print trainingExamples[0].label
  #print trainingExamples[0].features

  #leastSquare(trainingData,testData,trainingSize,testSize)
  #regularized(trainingData,testData,trainingSize,testSize,'l1')
  #regularized(trainingData,testData,trainingSize,testSize,'l2')
  decisionTreeRegression(trainingData,testData,trainingSize,testSize)

  sc.stop()







