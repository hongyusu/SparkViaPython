


from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import inspect
import logging

logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.WARNING)



def basic_statistics_and_plotting(lines):
  '''
  this function is designed to explore some basic statistics of the data
  input: RDD
  '''
  logging.warning('-------------------------- plotting basic statistics ---------------------')

  # -------------------------------- ckeck keyword = criterion id ------------------------------------------------
  # for now, keyword is defined as criterion_id, the following code is to validate if keyword is unique to adgroup
  keyword1 = lines.map( lambda x: (x[1].split(',')[1], [x[1].split(',')[0]]) ) \
                        .reduceByKey(lambda x,y: x+y) \
                        .map(lambda x: (x[0],[x[1],len(set(x[1]))]) )
  keyword2 = keyword1.filter(lambda x: x[1][1]>1)
  for item in keyword2.collect(): logging.info("%s %s %d" % (item[0],str(Counter(item[1][0])),item[1][1]))
  logging.warning('Percentage of keywords (criterion) that are not unique to adgroup :\t %.2f %%' \
                  % (float(keyword2.count()) / keyword1.count() * 100))

  # -------------------------------- ckeck keyword = criterion id + adgroup id------------------------------------------------
  # for now, keyword is defined as criterion_id:adgroup_id, the following code is to validate if keyword is unique to campaign 
  keyword1 = lines.map( lambda x: (x[1],[x[0]]) ) \
                        .reduceByKey(lambda x,y: x+y) \
                        .map(lambda x: (x[0],[x[1],len(set(x[1]))]) )
  keyword2 = keyword1.filter(lambda x: x[1][1]>1)
  for item in keyword2.collect(): logging.info("%s %s %d" % (item[0],str(Counter(item[1][0])),item[1][1]))
  logging.warning('Percentage of keywords (adgroup+criterion) that are not unique to campaign :\t %.2f %%' \
                  % (float(keyword2.count()) / keyword1.count() * 100))

  # -------------------------------- frequencies ------------------------------------------------
  campaignCount = lines.map( lambda x: ( x[0], 1) ) \
                      .reduceByKey(lambda x,y: x+y) \
                      .collect()
  keywordCount = lines.map( lambda x: ( x[1], 1) ) \
                      .reduceByKey(lambda x,y: x+y) \
                      .collect()
  dateCount = lines.map( lambda x: ( x[2], 1) ) \
                      .reduceByKey(lambda x,y: x+y) \
                      .collect()

  plt.figure(figsize=(25,20))

  subplot = plt.subplot(4,4,1)
  subplot.set_title('Campaign count')
  subplot.hist(np.array([x[1] for x in campaignCount]), bins=100)

  subplot = plt.subplot(4,4,2)
  subplot.set_title('Keyword count')
  subplot.hist(np.array([x[1] for x in keywordCount]), bins=100)

  subplot = plt.subplot(4,4,3)
  subplot.set_title('Date count')
  subplot.hist(np.array([x[1] for x in dateCount]), bins=100)


  # -------------------------------- date V.S. number of records -------------------------------
  tmpData = lines.map(lambda x: (x[2],1)).reduceByKey(lambda x,y:x+y) \
                 .collect()
  xs = [x[0] for x in tmpData]
  ys = [x[1] for x in tmpData]
  subplot = plt.subplot(4,4,4)
  subplot.scatter(xs,ys)
  subplot.set_xlabel('Day')
  subplot.set_ylabel('Number of record')

  # -------------------------------- disbribution of record -------------------------------
  tmpData = lines.collect()
  logging.warning("Number of campaign:\t %d" % len(set([x[0] for x in tmpData])))
  logging.warning("Number of keywords:\t %d" % len(set([x[1] for x in tmpData])))
  logging.warning("Number of date:\t %d"   % len(set([x[2] for x in tmpData])))

  # separate data
  xs = np.array([eval(x[0]) for x in tmpData])
  ys = np.array([eval(x[1].split(',')[1]) for x in tmpData])
  zs = np.array([x[2] for x in tmpData])

  # campaign vs date
  subplot = plt.subplot(4,4,5)
  subplot.scatter(xs,zs, marker='.', c='r')
  subplot.set_xlabel('Campaign')
  subplot.set_ylabel('Date')

  # keyword vs date
  subplot = plt.subplot(4,4,9)
  subplot.scatter(ys,zs, marker='.', c='r')
  subplot.set_xlabel('Keyword')
  subplot.set_ylabel('Date')

  # campaign vs keyword
  subplot = plt.subplot(4,4,13)
  subplot.scatter(xs,ys, marker='.', c='r')
  subplot.set_xlabel('Campaign')
  subplot.set_ylabel('Keyword')

  # campaign vs keyword vs date
  subplot = plt.subplot2grid((4,4),(1,1),colspan=3,rowspan=3,projection='3d')
  subplot.scatter(xs,ys,zs, marker='.', c='r') 
  subplot.set_xlabel('Campaign')
  subplot.set_ylabel('Keyword')
  subplot.set_zlabel('Date')
  plt.tight_layout()
  try:
    plt.savefig('./Results/%s.png' % (inspect.stack()[0][3]))  #plt.show()
  except:
    plt.savefig('./Results/basic_statistics_and_plotting.png')
  pass


def plot_global_as_time_1(lines):
  '''
  This function is designed to plot global situation (clicks, conversions, position) as a function of time
  '''
  logging.warning('-------------------------- plotting global 1 ---------------------')
  plt.figure(figsize=(16,6))
  interval = [0,2,4,100000]
  for j in range(len(interval)-1):
    for i,matchType in enumerate(['Exact','Phrase','Broad','all']):
      # collect data
      data = lines.filter(lambda x: ( x[4] == matchType or matchType=='all' ) and x[5] >= interval[j] and x[5]<interval[j+1]   ) \
                  .map(lambda x: (x[2],([x[3]],[x[5]],[x[6]])) ) \
                  .reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1],x[2]+y[2])) \
                  .map(lambda x: (x[0],( sum(x[1][0])/float(len(x[1][0])),sum(x[1][1])/float(len(x[1][1])),sum(x[1][2])/float(len(x[1][2]))  ))) \
                  .collect()
      npdata = np.array([ (item[0],item[1][0],item[1][1],item[1][2]) for item in data ])
      npdata = npdata[npdata[:,0].argsort()]
      # plot
      subplot = plt.subplot(3,4,j*4+1+i)
      subplot.plot(npdata[:,0],npdata[:,1],'-*',label='position')
      subplot.plot(npdata[:,0],npdata[:,2],'-*',label='click')
      subplot.plot(npdata[:,0],npdata[:,3],'-*',label='conversion')
      subplot.plot(npdata[:,0],npdata[:,3]/npdata[:,2]*100.0,'-*',label='rate')
      subplot.set_title("%s [%d,%d)" %(matchType,interval[j],interval[j+1]))
      subplot.set_xlabel('Time')
      subplot.set_ylabel('Clk / Cnv / Pos')
      subplot.grid(True)
      if i==0 and j==0:
        legend = subplot.legend(loc='upper right',shadow=False,prop={'size':3})
        frame = legend.get_frame()
        frame.set_facecolor('1')
        for label in legend.get_texts(): label.set_fontsize('small')
  plt.tight_layout()
  try:
    plt.savefig('./Results/%s.png' % (inspect.stack()[0][3]))  #plt.show()
  except:
    plt.savefig('./Results/plot_global_as_time_1.png')
  pass

def plot_global_as_time_2(lines):
  '''
  This function is designed to plot global situation (clicks, conversions, position) as a function of time
  '''
  logging.warning('-------------------------- plotting global 2 ---------------------')
  plt.figure(figsize=(16,6))
  interval = [0,2,4,100000]

  for j in range(len(interval)-1):

    # collect data
    data = lines.filter(lambda x: x[5] > interval[j] and x[5]<=interval[j+1]   ) \
                .map(lambda x: (x[2],([x[3]],[x[5]],[x[6]])) ) \
                .reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1],x[2]+y[2])) \
                .map(lambda x: (x[0],( sum(x[1][0])/float(len(x[1][0])),sum(x[1][1])/float(len(x[1][1])),sum(x[1][2])/float(len(x[1][2]))  ))) \
                .collect()
    npdata = np.array([ (item[0],item[1][0],item[1][1],item[1][2]) for item in data ])
    npdata = npdata[npdata[:,0].argsort()]

    # plot
    subplot = plt.subplot(3,4,j*4+1)
    subplot.plot(npdata[:,0],npdata[:,1],'-*',label='position')
    subplot.set_xlabel('Time')
    subplot.set_ylabel('Position')
    subplot.grid(True)
    subplot.set_title('[%d, %d)' % (interval[j],interval[j+1]))

    subplot = plt.subplot(3,4,j*4+2)
    subplot.plot(npdata[:,0],npdata[:,2],'-*',label='click')
    subplot.set_xlabel('Time')
    subplot.set_ylabel('Click')
    subplot.grid(True)
    subplot.set_title('[%d, %d)' % (interval[j],interval[j+1]))

    subplot = plt.subplot(3,4,j*4+3)
    subplot.plot(npdata[:,0],npdata[:,3],'-*',label='conversion')
    subplot.set_xlabel('Time')
    subplot.set_ylabel('Conversion')
    subplot.grid(True)
    subplot.set_title('[%d, %d)' % (interval[j],interval[j+1]))

    subplot = plt.subplot(3,4,j*4+4)
    subplot.plot(npdata[:,0],npdata[:,3]/npdata[:,2]*100.0,'-*',label='rate')
    subplot.set_xlabel('Time')
    subplot.set_ylabel('Rate')
    subplot.grid(True)
    subplot.set_title('[%d, %d)' % (interval[j],interval[j+1]))
  plt.tight_layout()
  try:
    plt.savefig('./Results/%s.png' % (inspect.stack()[0][3]))  #plt.show()
  except:
    plt.savefig('./Results/plot_global_as_time_2.png')
  pass


def mapper(line):
  '''
  mapper function
  '''
  r = []
  for item in line[1]:
    res = {}
    newres = []
    for a,b in item:
      if not a in res:
        res[a] = [b,1]
      else:
        res[a][0]+=b
        res[a][1]+=1
    for a in res.keys():
      newres.append( (a,res[a][0]/float(res[a][1])) )
    r.append(newres)
  return (line[0],r)
  pass

def plot_keyword_measure_days(lines):
  '''
  Average click/conversion/position of a keyword over days plotted according to time.
  Values are averaged over campaigns.
  Keywards are filter by the minimum data points it has during the period
  '''
  logging.warning('-------------------------- plotting local 1 ---------------------')
  plt.figure(figsize=(30,200))
  interval = [0,1,3,5,7,100000]
  minDataPoint = 2

  for j in range(len(interval)-1):

    # collect data from Spark
    data = lines.filter(lambda x: x[4]=='Broad') \
                .map(lambda x: (x[1],([(x[2],x[3])],[(x[2],x[5])],[(x[2],x[6])])) ) \
                .reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1],x[2]+y[2])) \
                .filter(lambda x: len(x[1][0])>=minDataPoint and max((np.array(x[1][1])[:,1]).tolist())>=interval[j] and max((np.array(x[1][1])[:,1]).tolist())<interval[j+1]  ) \
                .map(mapper) \
                .collect()
    # plot
    for i,item in enumerate(data):
      npdata = np.array(item[1][0])
      npdata = np.append(npdata,np.array(item[1][1]),axis=1)
      npdata = np.append(npdata,np.array(item[1][2]),axis=1)
      npdata = npdata[npdata[:,0].argsort()]

      subplot = plt.subplot(100,5,i*5+j+1)
      subplot.plot(npdata[:,0],npdata[:,1],'-*',label='position')
      subplot.plot(npdata[:,0],npdata[:,3],'-*',label='click')
      subplot.plot(npdata[:,0],npdata[:,5],'-*',label='conversion')
      #subplot.plot(npdata[:,0],npdata[:,5]/npdata[:,3]*50.0,'-*',label='rate')
      subplot.set_xlabel('Time %s' % item[0])
      subplot.set_ylabel('Clk / Cnv / Pos')
      subplot.set_title('max click [%d, %d) min point %d' % (interval[j],interval[j+1],minDataPoint))
      subplot.grid(True)
      legend = subplot.legend(loc='upper right',shadow=False,prop={'size':3})
      frame = legend.get_frame()
      frame.set_facecolor('1')
      for label in legend.get_texts(): label.set_fontsize('small')
      if i == 49: break

  #plt.tight_layout()
  try:
    plt.savefig('./Results/%s.png' % inspect.stack()[0][3])
  except:
    plt.savefig('./Results/plot_keyword_measure_days.png')
  pass


def plot_keyword_measure_weekdays(lines):
  '''
  Average click/conversion/position of a keyword over weekdays plotted according to time.
  Values are averaged over campaigns and weekdays.
  Keywards are filter by the minimum data points it has during the period
  '''
  logging.warning('-------------------------- plotting local 2 ---------------------')
  plt.figure(figsize=(30,200))
  interval = [0,1,3,5,7,100000]
  minDataPoint = 2

  for j in range(len(interval)-1):

    # collect data from Spark
    data = lines.map(lambda x: (x[1],([(x[2]%7,x[3])],[(x[2]%7,x[5])],[(x[2]%7,x[6])])) ) \
                .reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1],x[2]+y[2])) \
                .filter(lambda x: len(x[1][0])>=minDataPoint and max((np.array(x[1][1])[:,1]).tolist())>=interval[j] and max((np.array(x[1][1])[:,1]).tolist())<interval[j+1]  ) \
                .map(mapper) \
                .collect()
    # plot
    for i,item in enumerate(data):
      npdata = np.array(item[1][0])
      npdata = np.append(npdata,np.array(item[1][1]),axis=1)
      npdata = np.append(npdata,np.array(item[1][2]),axis=1)
      npdata = npdata[npdata[:,0].argsort()]

      subplot = plt.subplot(100,5,i*5+j+1)
      subplot.plot(npdata[:,0],npdata[:,1],'-*',label='position')
      subplot.plot(npdata[:,0],npdata[:,3],'-*',label='click')
      subplot.plot(npdata[:,0],npdata[:,5],'-*',label='conversion')
      #subplot.plot(npdata[:,0],npdata[:,5]/npdata[:,3]*50.0,'-*',label='rate')
      subplot.set_xlabel('Time %s' % item[0])
      subplot.set_ylabel('Clk / Cnv / Pos')
      subplot.set_title('max click [%d, %d) min point %d' % (interval[j],interval[j+1],minDataPoint))
      subplot.grid(True)
      legend = subplot.legend(loc='upper right',shadow=False,prop={'size':3})
      frame = legend.get_frame()
      frame.set_facecolor('1')
      for label in legend.get_texts(): label.set_fontsize('small')
      if i == 49: break

  #plt.tight_layout()
  try:
    plt.savefig('./Results/%s.png' % inspect.stack()[0][3])
  except:
    plt.savefig('./Results/plot_keyword_measure_weekdays.png')
  pass



#------------------------------------------------ plot missing value matrix -------------------------------------------
def mapper_plot_keyword_day_missing_value_matrix(line):
  '''
  mapper function for plot_keyword_day_matrix
  '''
  r = []
  for item in line[1]:
    res = {}
    newres = []
    for a,b in item:
      if not a in res:
        res[a] = [b,1]
      else:
        res[a][0]+=b
        res[a][1]+=1
    for a in range(6):
      if a in res.keys():
        newres.append( (a,res[a][0]/float(res[a][1])) )
      else:
        newres.append( (a,None) )
    r.append(newres)
  return (line[0],r)
  pass

def plot_keyword_day_missing_value_matrix(lines):
  '''
  the function plots keyword/day matrix to find out the amount of missing values
  '''
  plt.figure(figsize=(10,10))

  for j,matchType in enumerate(['Exact','Broad','Phrase','All']):
    # collect data from Spark
    data = lines.map(lambda x: (x[1],([(x[2],x[3])],[(x[2],x[5])],[(x[2],x[6])])) \
                               if x[4]==matchType or matchType=='All' else (x[1],([],[],[]))) \
              .reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1],x[2]+y[2])) \
              .map(mapper_plot_keyword_day_missing_value_matrix) \
              .collect()
    res = np.array([[ b for a,b in item[1][0]] for i,item in enumerate(data)],dtype=np.float)
    resKeyword = np.nansum(res,axis=1)
    # plot
    subplot = plt.subplot(1,4,j+1)
    subplot.imshow(res,aspect='auto',interpolation='nearest')
    subplot.set_xlabel('Time')
    subplot.set_ylabel('Keyword %d / %d' %(np.sum(np.isnan(resKeyword)) , float(resKeyword.shape[0])))
    subplot.set_title('%s %.2f %% ' % ( matchType,100-100*np.sum(np.isnan(res))/float(res.shape[0]*res.shape[1]) ) )
  plt.tight_layout()
  try:
    plt.savefig('./Results/plot_keyword_day_missing_value_matrix.png')
  except:
    plt.savefig('./Results/%s.png' % inspect.stack()[0][3])
  pass


#------------------------------------------------ plot imputation -------------------------------------------
def plot_imputation(data,name):
  '''
  plot imputation result
  '''
  logging.warning('-------------------------- plotting imputation results: %s ---------------------' % name)

  # plot
  plt.figure(figsize=(15,40))
  for i,item in enumerate(data):
    npdata = np.array(item[1])
    npdata = npdata[npdata[:,0].argsort()]
    subplot = plt.subplot(20,3,i+1)
    subplot.plot(npdata[:,0],npdata[:,1],'-o')
    for x in npdata[npdata[:,2] ==1][:,0]:
      plt.axvline( x )
    subplot.set_xlabel('Time')
    subplot.set_ylabel(name)
    subplot.set_title(item[0])
    if i==59: break
  plt.tight_layout()
  try:
    plt.savefig('./Results/plot_imputation_%s.png' % name)
  except:
    plt.savefig('./Results/%s_%s.png' % (inspect.stack()[0][3],name))
  pass

#------------------------------------------------ plot regression -------------------------------------------
def plot_regression(data):
  '''
  plot regression rmse
  '''
  logging.warning('-------------------------- plotting regression RMSE ---------------------')
  npdata = np.array(data)
  plt.figure(figsize=(7,7))
  subplot = plt.subplot(111)
  subplot.plot(npdata[:,1],'r-*',label='click')
  subplot.plot(npdata[:,2],'b-*',label='conversion')
  subplot.set_xlabel('keyword')
  subplot.set_ylabel('RMSE')
  legend = subplot.legend(loc='upper right',shadow=False,prop={'size':3})
  frame = legend.get_frame()
  frame.set_facecolor('1')
  for label in legend.get_texts(): label.set_fontsize('small')
  plt.tight_layout()
  try:
    plt.savefig('./Results/plot_regression.png')
  except:
    plt.savefig('./Results/%s.png' % inspect.stack()[0][3])
  pass

