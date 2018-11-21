# coding: utf-8

import gc
import time
import yaml
from utils import *
import numpy as np
import pandas as pd

import pickle
import logging
import logging.config
import os

import keras
from keras import objectives
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers import Bidirectional
from keras.layers import Input, LSTM, GRU, TimeDistributed, RepeatVector, Reshape, Permute
from keras.layers import Input, Dense, MaxPooling1D, Flatten, UpSampling1D, Lambda
from keras.layers import LeakyReLU, ELU, ThresholdedReLU, PReLU
from keras.models import Model, Sequential
from keras.regularizers import l1, l2
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras import metrics
from keras.optimizers import Adam

from hyperopt import Trials, STATUS_OK, tpe, rand
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

import tensorflow as tf
from keras import backend as K

with open('logging.yaml', 'r') as f:
	config = yaml.load(f)

logging.config.dictConfig(config)

logging.info('Load dataset')

with open('../data/train.pkl', 'rb') as f:
	train = pickle.load(f)


with open('../data/test.pkl', 'rb') as f:
	test = pickle.load(f)

# 무제한 로그 제외

train = train.replace([np.inf, -np.inf], np.nan).dropna(how= 'any')
test = test.replace([np.inf, -np.inf], np.nan).dropna(how= 'any')

# Unused col 제외
unused_cols = ['UUID', 'TRCR_NO', 'UNITRULE_CODE']
y_train = train['UNITRULE_CODE']
X_train = train.drop(unused_cols, axis = 1)

y_test = test["UNITRULE_CODE"]
X_test = test.drop(unused_cols, axis =1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

with open('../data/X_train.pkl', "wb") as f:
	pickle.dump(X_train, f, protocol = 4)

with open('../data/y_train.pkl', "wb") as f:
	pickle.dump(y_train, f, protocol = 4)

with open('../data/X_test.pkl', "wb") as f:
	pickle.dump(X_test, f, protocol = 4)

with open('../data/y_test.pkl', "wb") as f:
	pickle.dump(y_test, f, protocol = 4)

def data():
	with open('../data/X_train.pkl', 'rb') as f:
		X_train = pickle.load(f)

	with open('../data/y_train.pkl', 'rb') as f:
		y_train = pickle.load(f)

	return X_train, y_train, X_test, y_test

def create_model(X_train, y_train, X_test, y_test):
	def auc(y_ture, y_pred):
		auc = tf.metrics.auc(y_true, y_pred)[1]
		K.get_session().run(tf.local_variables_initializer())
		return auc

# Create model architecture
	model = Sequential()
	model.add(Dense({{choice([32,60,64,80,128])}}, activation={{choice(["relu", "tanh", "linear", "elu"])}}, input_dim = X_train.shape[1]))
	model.add(Dense({{choice([32,60,64,80,128])}}, activation={{choice(["relu", "tanh", "linear", "elu"])}}))
	model.add(Dense({{choice([32,60,64,80,128])}}, activation={{choice(["relu", "tanh", "linear", "elu"])}}))
	model.add(Dense({{choice([32,60,64,80,128])}}, activation={{choice(["relu", "tanh", "linear", "elu"])}}))
	model.add(Dense({{choice([32,60,64,80,128])}}, activation={{choice(["relu", "tanh", "linear", "elu"])}}))
	model.add(Dense(1, activation = "sigmoid"))
	print(model.summary())

	# Comliling
	adam = Adam(lr = 0.0001 , decay = 1e-6)
	model.compile(loss = "binary_crossentropy", optimizer = adam, metrics = ["acc", auc])

	# Training
	result = model.fit(X_train, y_train,
		epochs = 20
		batch_size = {{choice[64,128]}},
		validation_split = 0.2,
		shuffle = True)

	# Save the best result
	val_auc = np.amax(result.history["val_auc"])
	print(f"Best AUC : {val_auc}")
	return {"loss :" -val_auc, "status": STATUS_OK, "model": model}


## Searching

best_run, best_model = optim.minimize(model = create_model,
									data = data,
									algo = tpe.suggest, 
									max_evals = 100,
									trials = Trials())
from keras.models import save_model
save_model(best_model, 'best_model.h5')
print(best_model.to_json())