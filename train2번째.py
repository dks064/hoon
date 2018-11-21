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

# 예측 모델 파라미터
hist.params


y_pred = model.predict(X_test, batch_size=256, verbose = 1)
# 배치사이즈가 256인건 모델 핏 할때 64여서 , 0.2 벨리데이션 스플릿이 4배임.

y_pred = y_pred.flatten()
recons_err = y_pred

# 시각화

def visualize_anomaly(error_df, threshold = None):

	if threshold is None:
		threshold = error_df[error_df["Class"]
							== 1].reconstruction_error.quantile(q=0.95) # 95%보다 높게
	print("Generated threshold: {}".format(threshold))
	print(error_df[error_df['reconstruction_error'] > threshold].shape)
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

visualiza_anomaly(error_df)

def get_anomaly_score(test, pred):
	#return np.sqrt(mean_squared_error(test, pred)) # root mean squared error
	return mean_squared_error(test,pred)`

threshold = 0.5
def to_label(x, threshold):
	if x.reconstruction_error >= threshold:
		return 1
	return 0

error_df.columns

error_df = error_df.assign(y_pred_class = lambda x: x.reconstruction_error >= threshold)
error_df.y_pred_class = error_df.y_pred_class.replace({False : 0, True: 1})
confusion_matirx(error_df["Class"], error_df["y_pred_class"])

tn, fp, fn, tp = confusion_matrix(error_df["Class"], error_df["y_pred_class"]).ravel()
(tn, fp, fn, tp)

import math

mcc = 1.0 * (tp*tn - fp*fn)/math.sqrt((tp+fp)*(fp+fn)*(tn+fp)*(tn+fn))
mcc

#model 저장
from keras.model import save_model
save_model(model, 'm일자.h5')

# 에러율 저장
error_df = pd.DataFrame({"reconstruction_error" : recons_err, "Class" : y_test})
error_df.to_csv("recons_err.csv")

error_df.head(5)

# confusion matrixs
from sklearn.metrics import confusion_matrix, roc_curve, auc
fpr, tpr, threshold = roc_curve(error_df["Class"], error_df["reconstruction_error"], pos_label =2)
roc_auc = auc(fpr, tpr)
roc_auc
confusion_matrix(error_df["Class"], error_df["reconstruction_error"] > auc(fpr,tpr))


from sklearn.metrics import roc_auc_score, accurancy_score
roc_auc_score(error_df["Class"], error_df["reconstruction_error"])

#ROC 곡선
plt.plot(fpr, tpr, "b", label = "AUC = %0.2f" % roc_auc)
plt.title("ROC Curve")
plt.legend(loc = "best")
plt.xlim([0:1])
plt.ylim([0:1])
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")

# precision _ recall curve
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature

precision, recall, thresholds = precision_recall_curve(error_df["Class"], error_df["reconstruction_error"])

step_kwargs = ({"stpe" : "post"}
	if "step" in signature(plt.fill_between).parameter else {})
plt.step(recall, precision, color = "b", alpha =0.2, where = "post")
plt.fill_between(recall, precision, alpha = 0.2, color = "b", **step_kwargs)

plt.xlabel("Recall")
plt.ylabel("precision")
plt.xlim([0.0:1.05])
plt.ylim([0.0:1.0])
plt.title("Precision - Recall Curve : AP = {0:0.2f}".format(average_precision))

# precision recall 값
from sklearn.metrics import precision_score, recall_score
idx = len(recall[recalll > 0.8])
threshold = thresholds[idx]
y_pred_thre = (error_df["reconstruction_error"] > threshold)
print(threshold)
print(y_pred_thre[y_pred_thre == True].count())
print(precision_score(error_df["Class"], y_pred_thre))
print(recall_score(error_df["Class"], y_pred_thre))

# 확인 -> 0.3 이상 등등 확인
y_pred_thre = (error_df["reconstruction_error"] > 0.3)
print(threshold)
print(y_pred_thre[y_pred_thre == True].count())
print(precision_score(error_df["Class"], y_pred_thre))
print(recall_score(error_df["Class"], y_pred_thre))

error_df[()]

# 파일 저장
save_file = os.path.join(RESULT_PATH, "result.csv")
error_df.to_csv(save_file, index=True)

# 찾은 아이들 찾기
import pickle
filepath = os.path.join(BACKUP_PATH, "full_dataset.pkl")
with open(filepath, "rb") as f:
	origin = pickle.load(f)

total = error_df[error_df.reconstruction_error > 0.5]['reconstruction_error'].merge(origin[["UUID", "TRCR_NO", "TRD_DTM". "UNITRULE_CODE"]],
	right_index = True, left_index = True, how = "left")
total.UNITRULE_CODE.replace({np.NaN:0}, inplace = True)
total.sort_values(["TRCR_NO", "TRD_DTM"], ascending = True, inplace = True)
total = total[["UUID", "TRCR_NO", "TRD_DTM", "reconstruction_error", "UNITRULE_CODE"]]

# 저장
save_file = os.path.join(RESULT_PATH, "report_recon_err.csv")
total.to_csv(save_file,index = True)