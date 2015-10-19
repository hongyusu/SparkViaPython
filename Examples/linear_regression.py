

from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionModel
from pyspark.mllib.regression import  LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
import itertools
import math

  
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

def lasso(trainingData,testData,trainingSize,testSize):
  '''
  Least square with l1 norm: lasso
  '''
  # train a lr model
  numIterValList = [1000,3000,5000]
  stepSizeValList = [1e-11,1e-9,1e-7,1e-5]
  regParamValList = [0.01,0.1,1,10,100]

  # variable for the best parameters
  bestNumIterVal = 200
  bestStepSizeVal = 1
  bestTrainingRMSE = 1e10 
  bestRegParamVal = 0.0

  regTypeVal = 'l1'

  for numIterVal,stepSizeVal,regParamVal in itertools.product(numIterValList,stepSizeValList,regParamValList):
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



if __name__ == '__main__':

  # set up Spark environment
  APP_NAME = "Spark linear regression models"
  conf = SparkConf().setAppName(APP_NAME)
  conf = conf.setMaster('spark://ukko160:7077')
  sc = SparkContext(conf=conf)
  
  # load data from file
  #parsedData = MLUtils.loadLibSVMFile(sc, "../spark-1.4.1-bin-hadoop2.6/data/mllib/sample_libsvm_data.txt")
  parsedData = MLUtils.loadLibSVMFile(sc, "../Data/cadata")

  # split data into training and test
  trainingData,testData = parsedData.randomSplit([0.8,0.2])
  trainingSize = trainingData.count()
  testSize = testData.count()
  print "Training:\t%d\nTest:\t%d" % (trainingSize,testSize)

  #trainingExamples = trainingData.collect()
  #testExamples = testData.collect()
  #print trainingExamples[0].label
  #print trainingExamples[0].features

  #leastSquare(trainingData,testData,trainingSize,testSize)
  lasso(trainingData,testData,trainingSize,testSize)
  #ridgeRegression(trainingData,testData,trainingSize,testSize)

  sc.stop()







