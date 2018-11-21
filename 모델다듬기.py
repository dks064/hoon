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

logging.info('Load dataset')
with open('../data/X_train.pkl', 'rb') as f:
	X_train = pickle.load(f)


with open('../data/y_train.pkl', 'rb') as f:
	y_train = pickle.load(f)


with open('../data/X_test.pkl', 'rb') as f:
	X_test = pickle.load(f)


with open('../data/y_test.pkl', 'rb') as f:
	y_train = pickle.load(f)

logging.info('Done Loading dataset')

import tensorflow as tf
from keras import backend as K
def auc(y_true, y_pred):
	auc = tf.metrics.auc(y_true, y_pred)[1]
	K.get_session().run(tf.local_variables_initializer())
	return auc

logging.info('Load model')
from keras.models import load_model
from keras.optimizers import Adam
model = load_model('best_model.h5', custom_object = { 'auc' : auc})
logging.info(model.summary())
adam = Adam(lr = 0.0001, decay = 1e-6)
model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['acc', auc])

logging.info('Training')
result = model.fit(X_train, y_train,
					epochs = 100,
					batch_size = 64,
					validation_split = 0.2,
					shuffle = True)

logging.info('Save model')
model.save(__file__+'.h5')

logging.info('predicting')
y_pred = model.predict(X_test, verbose = 1, batch_size = 256)

y_pred = y_pred.flatten()

#recons_err = np.mean(np.power(X_test - X_pred, 2))
error_df = pd.DataFrame({'recons_err' : y_pred, 'Class' : y_test})
error_df.to_csv('recons_err')

Visualize_amonaly(error_df)