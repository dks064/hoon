import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append("/sorc001/gdssvc01/notebook/code/sup_model_1")
from config import *


# train test 불러오기
train = load_plk(os.path.join(BACKUP_PATH, "train.plk"), "rb")
test = load_plk(os.path.join(BACKUP_PATH, "test.plk"), "rb")

#빈칸 날리기 - log 무한대
train = train.replace([np.inf, -np.inf], np.nan).dropna(how = "any")
test = test.replace([np.inf, -np.inf], np.nan).dropna(how = "any")

# column 확인
train.columns.tolist()

# 사용하지 않는 columns
unused_cols = ["UUID", "TRCR_NO", "UNITRULE_CODE"]

# train / test set
y_train = train["UNITRULE_CODE"]
X_train = train.drop(unused_cols, axis =1)

y_test = test["UNITRULE_CODE"]
X_test = test.drop(unused_cols, axis =1)

for col in X_train.columns:
	print(X_train[col].describe())

# 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 스케일러 백업
import pickle
with open(os.path.join(BACKUP_PATH, "scaler.plk"), "wb") as f:
	pickle.dump(scaler, f, protocol = 4)

# Normal training
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
from keras.optimizers import Adam

# 가중치
from sklearn.utils import class_weight
cw = class_weigth.compute_class_weight('balanced', np.unique(y_train), y_train)
sw = class_weigth.compute_sample_weight('balanced', y_train)
cw, sw

# fraud
from sklearn.utils import shuffle

# tensorflow 시작
import tensorflow as tf
from keras import backend as K
def auc(y_true, y_pred):
	auc = tf.metrics.auc(y_true, y_pred)[1]
	K.get_session().run(tf.local_variables_initializer())
	return auc

# DNN
def create_dnn_model(input_dim):
	model = Sequential()
	model.add(Dense(64, activation = "relu", input_dim = input_dim))
	model.add(Dense(64, activation = "relu"))
	model.add(Dense(64, activation = "relu"))
	model.add(Dense(1, activation = "sigmoid"))
	print(model.summary())
	return model
model = create_dnn_model(X_train.shape[1])

# adam으로 손실함수 최소화
adam = Adam(lr = 0.0001, decay=1e-6)
model.compile(loss = "binary_crossentropy", optimizer = adam, metrics=["acc",auc])


#모델 핏

hist = model.fit(X_train, y_train,
	epochs = 10,
	batch_size = 64,
	validation_split = 0.2,
	shuffle = True
	)

hist.history

y_pred = model.predict(X_test, batch_size=256, verbose = 1)
# 배치사이즈가 256인건 모델 핏 할때 64여서 , 0.2 벨리데이션 스플릿이 4배임.

y_pred = y_pred.flatten()
recons_err = y_pred

# 시각화

def visualize_anomaly(error_df, threshold = None):

	if threshold is None:
		threshold = error_df[error_df["Class"]
							== 1].reconstruction_error.quantile(q=0.95) # 90%보다 높게
	print("Generated threshold: {}".format(threshold))
	groups = error_df.groupby("Class")
	fig, ax = plt.subplots(figsize = (20,12))
# 		cm = sns.color_palette("RdBu_r", as_cmap=True)
	cmap = sns.cubehelix_palette(rot=-.4, reverse = True, as_cmap = True)
	for name, group in groups:
		ax.plot(group.index, group.reconstruction_error, marker="o", linestyle = "", alpha = 0.6,
			label = "Fraud" if name ==1 else "Normal",
			color = "r", if name ==1 else "royalblue")
	ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1],
			colors ="r", zorder =100, label="Threshold")
	ax.legend()

	plt.title("Anomaly Visualization")
	plt.xlabel("Number of Samples")
	plt.ylabel("Anomaly Score")

	plt.savefig("anomaly.png")
	plt.show()

# 에러율 저장
error_df = pd.DataFrame({"reconstruction_error" : recons_err, "Class" : y_test})
error_df.to_csv("recons_err.csv")

error_df.head(5)

# confusion matrixs
from sklearn.metrics import confusion_matrix, roc_curve, auc
fpr, tpr, threshold = roc_curve(error_df["Class"], error_df["reconstruction_error"], pos_label =2)
confusion_matrix(error_df["Class"], error_df["reconstruction_error"] > auc(fpr,tpr))

visualiza_anomaly(error_df)

error_df[()]