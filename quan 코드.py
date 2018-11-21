import pickle
import pandas as pd
import numpy as np
import os

import sys
sys.path.append('/sorc001/gdssvc01/notebook/code/sup_model_1')
from config import *

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# data 불러오기
frame = load_plk(os.path.join(BACKUP_PATH,'preprocessed_data_set.pkl'), 'rb')
frame.shpae, frame.info()

# data 형태 보기
amt_cols = ["BFTD_BAL","CARD_TRD_AMT", "AFT_TRD_BAL","log_BFTD_BAL","log_CARD_TRD_AMT","log_AFT_TRD_BAL"]

groups = frame.groupby("UNITRULE_CODE")
for col in amt_cols:
	fig, ax = plt.subplots()
	for name, group in groups:
		sns.kdeplot(np.log10(group[col]), shade = True, color = "r" , if name==1 else "g", ax=ax)

# train set 찢기
y = frame["UNITRULE_CODE"]
X = frame.drop("UNITRULE_CODE", axis = 1)

# 싸이킷 런 불러와서 랜덤 스플릿
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33 , random_state = 42)


train = pd.concat([X_train, y_train], axis = 1)
test = pd.concat([X_test, y_test], axis = 1)

train.UNITRULE_CODE.value_counts()
test.UNITRULE_CODE.value_counts()

# heatmap 
fig, ax = plt.subplots(figsize = (20,10))
sns.heatmap(frame.corr(), ax=ax)


# 저장
with open(os.path.join(BACKUP_PATH, 'train.plk'), "wb") as f:
	pickle.dump(train, f, protocol = 4)

with open(os.path.join(BACKUP_PATH, 'test.plk'), "wb") as f:
	pickle.dump(test, f, protocol = 4)
