import gc
import time
import yaml
from utils import *
import numpy as np
import pandas as pd

import pickle
import logging
import logging.config
import keras
import tensorflow as tf
from keras import objectives
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers import Bidirectional
from keras.layers import Input, LSTM, GRU, TimeDistributed, RepeatVector, Reshape, Permute
from keras.layers import Input, Dense, MaxPooling1D, Flatten, Upsampling1D, Lambda
from keras.models import model, Sequential
from keras.regularizers import l1, l2
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras import metrics
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

with open('logging.yaml', 'r') as f:
	config = yaml.load(f)

logging.config.dictConfig(config)

RANDOM_STATE = 42
BATCH_SIZE = 128

logging.info('Load dataset')
with open('../data/train.pkl', 'rb') as f:
	train = pickle.load(f)

logging.info('Done Loading dataset')

logging.info('vectorize')
dummy_list = train.LDGR_TRD_DVS_CD.unique().tolist()
train.LDGR_TRD_DVS_CD = pd.Categorical(train.LDGR_TRD_DVS_CD, categories = dummy_list)
dummy = pd.get_dummies(train.LDGR_TRD_DVS_CD, prefix = 'LDGR_TRD_DVS_CD', dummy_na = True)
train = pd.concat([train, dummy], axis = 1)
train = train.drop('LDGR_TRD_DVS_CD', axis= 1)

with open('../data/train_new.pkl', 'wb') as f:
	pickle.dump(train, f, protocol = 4)

logging.info('Extract fraud data')
# fraud = train[train['UNITRULE_CODE'] ==1]
train = train[train['UNITRULE_CODE'] ==0]

logging.info('Sampling')
train = train.sample(frac = 0.5, random_state = 42)
# train = train.append(fraud)

logging.info(train.shape)
logging.info('Create train / test dataset')
y_train = train['UNITRULE_CODE']
X_train = train.drop(unused_cols, axis = 1)

del dummy
del train

## Modeling

logging.info(X_train.info())

# In[]:
gc.collect()
logging.info('Normalize')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(copy = False)
# scalar.fit(X_train)
X_train = scaler.fit_transform(X_train)

with open('scaler.pkl', 'wb') as f:
	pickle.dump(scaler, f)

# logging.info('Compute_class_weight / sample_weight')
# from sklearn.util import class_weight
# cw = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
# sw = class_weight.compute_sample_weight('balanced', y_train)

# logging.info(f' Class weight : {cw})
# logging.info(f' Sample weight : {sw})

logging.info('Modeling')

cb = ModelCheckpoint('checkpoint/weights.{epoch:02d}-{val_loss:.2f}.h5', svae_best_only = True)
tb = TensorBoard(log_dir ='./logs', batch_size = BATCH_SIZE)
csvlogger = CSVLOGGER('run.log')

def create_dnn_model(input_dim):
	input_sample = Input(shape = (input_dim,))

	intermediate = 400

	encoded = Dense(intermediate , activation = 'relu')(input_sample)
	encoded = Dense(intermediate , activation = 'relu')(encoded)
	encoded = Dense(intermediate , activation = 'relu')(encoded)

	encoded = Dense(2 , activation = 'linear', name = 'latent')(encoded)

	encoded = Dense(intermediate , activation = 'relu')(encoded)
	encoded = Dense(intermediate , activation = 'relu')(encoded)
	encoded = Dense(intermediate , activation = 'relu')(encoded)

	dencoded = Dense(input_dim , activation = 'linear', name = 'fc')(dencoded)

	model = Model(input_sample, decoded)

	return model

#adam = tf.train.AdamOptimizer(0.001)
model = create_dnn_model(X_train.shape[1])
model.compile(optimizer = 'adam', loss = 'mse', mertics = ['accuracy'])

logging.info('Training')
hist = model.fit(X_train, X_train,
	epochs = 2,
	batch_size - BATCH_SIZE,
	shuffle = True,
	validation_split = 0.1,
	callbacks = [cb, tb, csvlogger],
	#validation_data = (X_val, X_val),
	verbose = 1)

logging.info('Save model')
model.save(__file__+'.h5')

del X_train
del y_train