

import itertools
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils

def svm(trainingData,testData,trainingSize,testSize):
  '''
  linear svm classifier
  '''
  # train a SVM model
  numIterValList = [100,200]
  regParamValList = [0.01,0.1,1,10,100]
  stepSizeValList = [0.1,0.5,1]
  regTypeValList = ['l2','l1']

  # variable for the best parameters
  bestNumIterVal = 200
  bestRegParamVal = 0.01
  bestStepSizeVal = 1
  bestRegTypeVal = 'l2'
  bestTrainErr = 100

  for numIterVal,regParamVal,stepSizeVal,regTypeVal in itertools.product(numIterValList,regParamValList,stepSizeValList,regTypeValList):
    break
    model = SVMWithSGD.train(trainingData, iterations=numIterVal, regParam=regParamVal, step=stepSizeVal, regType=regTypeVal)
    labelsAndPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(trainingSize)
    if trainErr<bestTrainErr:
      bestNumIterVal = numIterVal
      bestRegParamVal = regParamVal
      bestStepSizeVal = stepSizeVal
      bestRegTypeVal = regTypeVal
      bestTrainErr = trainErr
    print numIterVal,regParamVal,stepSizeVal,regTypeVal,trainErr
  print bestNumIterVal,bestRegParamVal,bestStepSizeVal,bestRegTypeVal,bestTrainErr

  model = SVMWithSGD.train(trainingData, iterations=bestNumIterVal, regParam=bestRegParamVal, step=bestStepSizeVal, regType=bestRegTypeVal)

  # Evaluating the model on training data
  labelsAndPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
  trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(trainingSize)
  print trainErr

  # Evaluating the model on training data
  labelsAndPreds = testData.map(lambda p: (p.label, model.predict(p.features)))
  testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(testSize)
  print testErr
  pass

  
def lr(trainingData,testData,trainingSize,testSize):
  '''
  linear lr classifier
  '''
  # train a lr model
  numIterValList = [100,200]
  regParamValList = [0.01,0.1,1,10,100]
  stepSizeValList = [0.1,0.5,1]
  regTypeValList = ['l2','l1']

  # variable for the best parameters
  bestNumIterVal = 200
  bestRegParamVal = 0.01
  bestStepSizeVal = 1
  bestRegTypeVal = 'l2'
  bestTrainErr = 100

  for numIterVal,regParamVal,stepSizeVal,regTypeVal in itertools.product(numIterValList,regParamValList,stepSizeValList,regTypeValList):
    model = LogisticRegressionWithSGD.train(trainingData, iterations=numIterVal, regParam=regParamVal, step=stepSizeVal, regType=regTypeVal)
    labelsAndPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(trainingSize)
    if trainErr<bestTrainErr:
      bestNumIterVal = numIterVal
      bestRegParamVal = regParamVal
      bestStepSizeVal = stepSizeVal
      bestRegTypeVal = regTypeVal
      bestTrainErr = trainErr
    print numIterVal,regParamVal,stepSizeVal,regTypeVal,trainErr
  print bestNumIterVal,bestRegParamVal,bestStepSizeVal,bestRegTypeVal,bestTrainErr

  model = LogisticRegressionWithSGD.train(trainingData, iterations=bestNumIterVal, regParam=bestRegParamVal, step=bestStepSizeVal, regType=bestRegTypeVal)

  # Evaluating the model on training data
  labelsAndPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
  trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(trainingSize)
  print trainErr

  # Evaluating the model on training data
  labelsAndPreds = testData.map(lambda p: (p.label, model.predict(p.features)))
  testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(testSize)
  print testErr
  pass



if __name__ == '__main__':

  # set up Spark environment
  APP_NAME = "Spark linear classification models"
  conf = SparkConf().setAppName(APP_NAME)
  conf = conf.setMaster('spark://ukko160:7077')
  sc = SparkContext(conf=conf)
  
  # load data from file
  parsedData = MLUtils.loadLibSVMFile(sc, "../spark-1.4.1-bin-hadoop2.6/data/mllib/sample_libsvm_data.txt")
  parsedData = MLUtils.loadLibSVMFile(sc, "../Data/a6a")
  #parsedData = MLUtils.loadLibSVMFile(sc, "../Data/gisette_scale")

  # split data into training and test
  trainingData,testData = parsedData.randomSplit([0.8,0.2])
  trainingSize = trainingData.count()
  testSize = testData.count()
  print "Training:\t%d\nTest:\t%d" % (trainingSize,testSize)
  trainingExamples = trainingData.collect()
  testExamples = testData.collect()
 
  #svm(trainingData,testData,trainingSize,testSize)
  lr(trainingData,testData,trainingSize,testSize)

  sc.stop()







