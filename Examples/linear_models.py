
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

  # split data into training and test
  train,test = parsedData.randomSplit([0.6,0.4])
  trainSize = train.count()
  testSize = test.count()
  print "Training:\t%d\nTest:\t%d" % (trainSize,testSize)
  
  # train a SVM model
  model = SVMWithSGD.train(train, iterations=100)
  
  # Evaluating the model on training data
  labelsAndPreds = train.map(lambda p: (p.label, model.predict(p.features)))
  trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / trainSize
  print trainErr

  # Evaluating the model on training data
  labelsAndPreds = test.map(lambda p: (p.label, model.predict(p.features)))
  testErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / testSize 
  print testErr

  sc.stop()
  


if __name__ == '__main__':
  svm()
