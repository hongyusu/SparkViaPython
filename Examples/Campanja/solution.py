
'''
solution for Campanja
'''


# python packages included in this scsript

from pyspark import SparkContext
from pyspark import SparkConf

from datetime import datetime
import logging
import plotting
import learning


logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.WARNING)

# meta information
__author__ = 'Hongyu Su'
__version__ = '0.1'



if __name__ == '__main__':
  logging.warning('--------------------------------- main --------------------------------- ')
  # configuration of spark engine
  conf = SparkConf().setAppName('Campanja')
  #conf = conf.setMaster('spark://ukko160:7077')
  conf = conf.setMaster('local[2]')
  sc = SparkContext(conf=conf)

  inputFilename = './campanja_work_sample_data_customerA.csv'
  # spark context
  lines = sc.textFile(inputFilename) \
            .filter(lambda x: not x.startswith('campaign')) \
            .map(lambda x: x.split(','))

  logging.warning('--------------------------------- preprocessing --------------------------------- ')
  # index date
  mydate = lines.map(lambda x: datetime.strptime(x[3],"%Y-%m-%d")) \
                .collect()
  firstDay = min(mydate)
  lastDay = max(mydate)
  logging.warning('First date:\t %s (day) %d' % (firstDay,(firstDay-firstDay).days))
  logging.warning('Last  date:\t %s (day) %d' % (lastDay, (lastDay-firstDay).days))
  lines = lines.map(lambda x: [x[0], x[1] + ',' + x[2], \
                               (datetime.strptime(str(x[3]),"%Y-%m-%d")-firstDay).days, \
                               eval(x[4]), x[5], eval(x[6]), eval(x[7])])

  plotting.basic_statistics_and_plotting(lines)

  # index keyword
  mykeyword = lines.map(lambda x: x[1]).distinct().collect()
  keywordMap = {item:i for i,item in enumerate(mykeyword)}
  keywordReverseMap = {i:item for i,item in enumerate(mykeyword)}
  logging.warning('Number of keywords:\t %d' % len(keywordMap.keys()))
  lines = lines.map(lambda x: [x[0],keywordMap[x[1]],x[2],x[3],x[4],x[5],x[6]])

  logging.warning('--------------------------------- data exploration --------------------------------- ')

  # plot all kinds of statistics of data
  plotting.plot_global_as_time_1(lines)
  plotting.plot_global_as_time_2(lines)
  plotting.plot_keyword_measure_days(lines)
  plotting.plot_keyword_measure_weekdays(lines)
  plotting.plot_keyword_day_missing_value_matrix(lines)

  logging.warning('--------------------------------- imputation --------------------------------- ')
  # collaborative filtering ALS for missing value imputation
  clickData      = learning.missing_value_imputation(lines,5)
  conversionData = learning.missing_value_imputation(lines,5)
  plotting.plot_imputation(clickData.collect(),      'click')
  plotting.plot_imputation(conversionData.collect(), 'conversion')

  logging.warning('--------------------------------- regression --------------------------------- ')
  # do regression locally for each keyword
  clickPrediction          = learning.local_regression(clickData,'click')
  conversionPrediction     = learning.local_regression(conversionData,'conversion')

  # collect rmse from each keyword and plot
  clickError      = {item[0]:item[1][1] for i,item in enumerate(clickPrediction.collect())}
  conversionError = {item[0]:item[1][1] for i,item in enumerate(conversionPrediction.collect())}
  for key in clickError.keys():
    clickError[key] = (key,clickError[key],conversionError[key])
  regressionError = clickError.values()
  plotting.plot_regression(regressionError)

  logging.warning('--------------------------------- output --------------------------------- ')
  # collect data
  clickPrediction      = {item[0]:item[1][0] for i,item in enumerate(clickPrediction.collect())}
  conversionPrediction = {item[0]:item[1][0] for i,item in enumerate(conversionPrediction.collect())}
  
  # write results to file
  fout = open('./Results/'+inputFilename+'.res','w')
  for key in clickPrediction.keys():
    try:
      fout.write("%s,%.2f\n" % (keywordReverseMap[key],max(0,conversionPrediction[key]/float(clickPrediction[key]))))
    except Exception as msg:
      logging.warning(msg)
      fout.write("%s,%.2f\n" % (keywordReverseMap[key],0))
    #print key,keywordReverseMap[key],conversionPrediction[key],clickPrediction[key]
  fout.close()

  sc.stop()





  
