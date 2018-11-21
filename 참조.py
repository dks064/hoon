# 패키지 불러오기

import pandas as pd 
import time 
import numpy as np 
import pickle 
import json 
from dateutil.parser import parse 
import datetime 
import seaborn as sns
import matplotlib.pyplot as plt

# 로깅 모듈
def logger_hander():
   import logging
   logger = logging.getLogger(__name__) 
   logger.setLevel(logging.INFO) 
   # create a file handler 
   handler = logging.FileHandler('preprocess.log') 
   handler.setLevel(logging.INFO) 
   # create a logging format 
   formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') #%(asctime)s - %(name)s - %(levelname)s - %(message)s 
   handler.setFormatter(formatter) 
   # add the handlers to the logger 
   logger.addHandler(handler) 
   return logger

## 백터라이징
def dummies(data, column, lists): 
  for col in lists: 
    colname = str(column)+'_'+str(col) 
    data[colname] = (data[column] == col) 
    data = data.drop(column , axis = 1) 
  return data 


## 임베딩
def get_corpus(data,embedding_col): 
  word_list = [] 
  temp = data.groupby('USR_ID')[embedding_col].agg(set) 

  for value in temp: 
    word_list.append(list(value)) 

  return corpus 


def get_embedded_feature(corpus): 
  from gensim.models import Word2Vec 
  from sklearn.decomposition import PCA 

  print('corpus created..') 

  w2v_model = Word2Vec(corpus, size=50, window=3, min_count=1, workers=4) 
  w2v_model.train(corpus, total_examples=len(corpus), epochs=10) 

  items = list(w2v_model.wv.vocab.keys()) 
  item_vectors = w2v_model.wv.vectors 

  n_comp = 3 
  pca_model = PCA(n_components=n_comp) 
  print("PCA components : {}".format(n_comp)) 

  trans = pca_model.fit_transform(item_vectors) 
  embedded_feature = {} 

  for item, vector in zip(items,trans) : 
    embedded_feature[item] = vector 

  return embedded_feature, w2v_model, pca_model 


# HOME
HOMEPATH = '~/notebook'

logger = logger_hander()

# config
DATA_DIR = "HOMEPATH/data"    
BACKUP_DIR = "HOMEPATH/backupdata/part1" 
RESULT_DIR = 'HOMEPATH/result/part1' 
logger.info("DATA_DIR [{}]".format(DATA_DIR)) 
logger.info("BACKUP_DIR [{}]".format(BACKUP_DIR)) 
logger.info("RESULT_DIR [{}]".format(RESULT_DIR)) 

# load_data
names = ["UUID","ID","COLCT_DTTM",
      "PROMOTION_AMOUNT","CLASS"] 
types = {
   'UUID':'str'
   ,'ID':'str'
   ,'COLCT_DTTM':'str'
   ,'PROMOTION_AMOUNT':np.float32
   ,"CLASS" : np.int32
}
dataset = pd.read_csv(os.pat.join(DATA_DIR, 'file_name.csv')
   , names = names
   , low_memory = False
   , dtype = types) 


vectorize_list = ['A','b']
col = 'CLASS'
dummies(dataset, , vectorize_list)
logger.info('{} : verctorize {}'.format(col, vectorize_list))