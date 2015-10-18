
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
  examples = parsedData.collect()
  print examples[0].label
  print examples[0].features
  print examples[0].features.size

  # train a SVM model
  model = SVMWithSGD.train(parsedData, iterations=100)
  
  # Evaluating the model on training data
  labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
  trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
  print("Training Error = " + str(trainErr))

  sc.stop()
  


if __name__ == '__main__':
  svm()
