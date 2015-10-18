

import itertools
from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils

def svm():
  '''
  linear svm classifier
  '''
  # set up Spark environment
  APP_NAME = "Collaboratove filtering for movie recommendation"
  conf = SparkConf().setAppName(APP_NAME)
  conf = conf.setMaster('spark://ukko160:7077')
  sc = SparkContext(conf=conf)
  
  # load data from file
  parsedData = MLUtils.loadLibSVMFile(sc, "../spark-1.4.1-bin-hadoop2.6/data/mllib/sample_libsvm_data.txt")
  parsedData = MLUtils.loadLibSVMFile(sc, "../Data/a6a")

  # split data into training and test
  trainingData,testData = parsedData.randomSplit([0.8,0.2])
  trainSize = trainingData.count()
  testSize = testData.count()
  print "Training:\t%d\nTest:\t%d" % (trainSize,testSize)
  trainingExamples = trainingData.collect()
  testExamples = testData.collect()
  print trainingExamples[0]
  print testExamples[0]
  
  
  # train a SVM model

  numIterValList = [100,200]
  regParamValList = [0.01,0.1,1,10]
  stepSizeValList = [0.1,0.5,1]
  regTypeValList = ['l2','l1']

  bestNumIterVal = 0
  bestRegParamVal = 0
  bestStepSizeVal = 0
  bestRegTypeVal = 0
  bestTrainErr = 100
  for numIterVal,regParamVal,stepSizeVal,regTypeVal in itertools.product(numIterValList,regParamValList,stepSizeValList,regTypeValList):
    model = SVMWithSGD.train(trainingData, iterations=numIterVal, regParam=regParamVal, step=stepSizeVal, regType=regTypeVal)
    labelsAndPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / trainSize
    if trainErr<bestTrainErr:
      bestNumIterVal = numIterVal
      bestRegParamVal = regParamVal
      bestStepSizeVal = stepSizeVal
      bestRegTypeVal = regTypeVal
      bestTrainErr = trainErr
    break
  print bestNumIterVal,bestRegParamVal,bestStepSizeVal,bestRegTypeVal,bestTrainErr

  # Evaluating the model on training data
  labelsAndPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
  trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / trainSize
  print trainErr,trainSize

  # Evaluating the model on training data
  labelsAndPreds = testData.map(lambda p: (p.label, model.predict(p.features)))
  testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / testSize 
  print testErr

  sc.stop()
  


if __name__ == '__main__':
  svm()
