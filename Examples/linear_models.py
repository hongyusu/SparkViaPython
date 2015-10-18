
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
  #parsedData = MLUtils.loadLibSVMFile(sc, "../Data/a6a")

  # split data into training and test
  trainingData,testData = parsedData.randomSplit([0.8,0.2])
  trainSize = trainingData.count()
  testSize = testData.count()
  print "Training:\t%d\nTest:\t%d" % (trainSize,testSize)
  
  # train a SVM model
  numIters = [100,200]
  regParas = [0.01,0.1,1,10,100]

  numIterVal = 100
  regParamVal = 0.1
  stepSizeVal = 0.5
  model = SVMWithSGD.train(trainingData, numIterations=numIterVal, regParam=regParamVal, stepSize=stepSizeVal)
  
  # Evaluating the model on training data
  labelsAndPreds = trainingData.map(lambda p: (p.label, model.predict(p.features)))
  trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / trainSize
  print trainErr

  # Evaluating the model on training data
  labelsAndPreds = testData.map(lambda p: (p.label, model.predict(p.features)))
  testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / testSize 
  print testErr

  sc.stop()
  


if __name__ == '__main__':
  svm()
