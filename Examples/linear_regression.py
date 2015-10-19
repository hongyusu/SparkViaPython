

from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionModel
from pyspark.mllib.regression import  LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
import itertools
import math

  
def linearRegression(trainingData,testData,trainingSize,testSize):
  '''
  linear lr classifier
  '''
  # train a lr model
  numIterValList = [100,200]
  stepSizeValList = [0.1,0.5,1]

  # variable for the best parameters
  bestNumIterVal = 200
  bestStepSizeVal = 1
  bestTrainingRMSE = 100

  regParamVal = 0.0
  regTypeVal = None

  for numIterVal,stepSizeVal in itertools.product(numIterValList,stepSizeValList):
    model = LinearRegressionWithSGD.train(trainingData, iterations=numIterVal, step=stepSizeVal, regParam=regParamVal, regType=regTypeVal)
    ValsandPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
    trainingRMSE = math.sqrt(ValsandPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / trainingSize)
    if trainingRMSE<bestTrainingRMSE:
      bestNumIterVal = numIterVal
      bestStepSizeVal = stepSizeVal
      bestTrainingRMSE = trainingRMSE
    print numIterVal,stepSizeVal,trainingRMSE
    break
  print bestNumIterVal,bestStepSizeVal,bestTrainingRMSE

  model = LinearRegressionWithSGD.train(trainingData, iterations=bestNumIterVal, step=bestStepSizeVal, regParam=regParamVal, regType=regTypeVal)

  # Evaluating the model on training data
  ValsandPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
  trainingRMSE = math.sqrt(valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / trainingSize)
  print trainingRMSE

  # Evaluating the model on training data
  ValsandPreds = testData.map(lambda p: (p.label, model.predict(p.features)))
  trainingRMSE = math.sqrt(valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / testSize)
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
  trainingExamples = trainingData.collect()
  testExamples = testData.collect()
 
  linearRegression(trainingData,testData,trainingSize,testSize)

  sc.stop()







