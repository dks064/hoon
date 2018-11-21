# 판다스와 넘파이, 그리고 옵션 세팅

import pandas as pd
import os
import numpy as np
import time
import sys
sys.path.append('/sorc001/gdssvc01/notebook/code/sup_model_1/')\
from config import *

start = time.process_time()
pd.set_option('display.max_columns' , 250)
pd.set_option('display.max_colwidth' , 200)
pd.set_option('display.max_seq_item' , 1000)

# 피클 불러와서 파일 내려받기
 import pickle
 
 filepath = os.path.join(BACKUP_PATH , 'full_dataset.pkl')

 with open(filepath , 'rb') as f:
	TBRLZM010 = pickle.load(f)

# 데이터 확인

TBRLZM010.shape
TBRLZM010.head()

# data rename
data.rename(columns = {'Hipertension': 'hypertension',
                       'Handcap': 'handicap',
                       'PatientId': 'patient_id',
                       'AppointmentID': 'appointment_id',
                       'ScheduledDay': 'scheduled_time',
                       'AppointmentDay': 'appointment_day',
                       'Neighbourhood': 'neighborhood',
                       'No-show': 'no_show'}, inplace = True)

참조염


# 8/4 이전 데이터만 추출
 TBRLZM010 = TBRLZM010[TBRLZM010.CLOS_DT <= "20180804"]
 TBRLZM010 = TBRLZM010[['UUID' , "TRCR_NO" , 'TRD_DTM' , 'LDGR_TRD_DTL_DVS_CD' , 'CLOS_DT' , 
 'LDGR_TBDS_DVS_CD' , 'BFTD_DTM' , 'BFTD_BAL' , 'CARD_TRD_AMT' , 'AFT_TRD_BAL' , 'LDGR_TRD_DVS_CD' , 
 'LDGR_ERR_CD' , 'TRLD_CARD_DVS_CD' , 'UNITRULE_CODE']]

 # 거래원장에서 이러한 데이터만 추출

 ## Filter
# F코드 다 빼버림
 TBRLZM010 = TBRLZM010[TBRLZM010.LDGR_TRD_DTL_DVS_CD.isin(['FSR','FSA','FS1','FBR','FBA','FB1','FTZ','FTC','FRN']) == False]
# MLZ 코드도 빼버림
TBRLZM010 = TBRLZM010[TBRLZM010.LDGR_TRD_DVS_CD != 'MLZ']
# 에러코드 1000 삭제
TBRLZM010 = TBRLZM010[(TBRLZM010.LDGR_ERR_CD != '1000')]
# 가맹점 ID 및 거래이후 시간 drop
TBRLZM010.drop('BFTD_DTM' , axis = 1 , inplace = True)
TBRLZM010.drop('LDGR_TBDS_DVS_CD' , axis = 1 , inplace = True)

## time format
TBRLZM010.TRD_DTM = TBRLZM010.TRD_DTM.apply(lambda x : date_parser(x,'%Y%m%d%H%M%S'))

## split datetime

TBRLZM010['trd_dtm_year'] = TBRLZM010.TRD_DTM.dt.year
TBRLZM010['trd_dtm_month'] = TBRLZM010.TRD_DTM.dt.month
TBRLZM010['trd_dtm_day'] = TBRLZM010.TRD_DTM.dt.day
TBRLZM010['trd_dtm_hour'] = TBRLZM010.TRD_DTM.dt.hour
TBRLZM010['trd_dtm_minute'] = TBRLZM010.TRD_DTM.dt.minute
TBRLZM010['trd_dtm_second'] = TBRLZM010.TRD_DTM.dt.second
TBRLZM010['trd_dtm_weeks'] = TBRLZM010.TRD_DTM.dt.dayofweek

# 잔액 미차감 차원

TBRLZM010 = TBRLZM010.sort_values('TRD_DTM' , ascending = True)
group_data = TBRLZM010.groupby("TRCR_NO" , as_index = False)["BFTD_BAL"].diff()

# week of day
TBRLZM010 = TBRLZM010.assign(trd_dtm_wom = TBRLZM010.TRD_DTM.apply(lambda x : week_of_month(x)))
TBRLZM010.trd_dtm_wom.value_counts(dropna = False)

## 참조  : 잔액 미차감 차원
TBRLZM010 = TBRLZM010.sort_values("TRD_DTM",ascending = True)
group_data = TBRLZM010.groupby("TRCR_NO", as_index=False)["BFTD_BAL"].diff()
TBRLZM010 = TBRLZM010.merge(pd.DataFrame(group_data).rename(columns = {"BFTD_BAL":"DIFF_BFTD_BAL"}), 
	left_index = True,right_index = True)
TBRLZM010 = TBRLZM010.assign(DIFF_BFTD_BAL_YN = (TBRLZM010.DIFF_BFTD_BAL == 0) *1 )


# Vertorize
def vectorize(frame, col, value):
	dummy_list = value #frame.col.unique().tolist()
	frame[col] = pd.Categorical(frame[col], categories = dummy_list)
	dummy = pd.get_dummies(frame[col], dummy_na = True)
	fram = pd.concat([frame , dummy], axis = 1)
	frame = frame.drop(col, axis =1)

# 주중 나누기
TBRLZM010.loc[TBRLZM010["trd_dtm_weeks"].isin([5,6]),"timework"] = 0
TBRLZM010.loc[TBRLZM010["trd_dtm_weeks"].isin([0:4]),"timework"] = 1

# 코드 쪼개기
TBRLZM010 = TBRLZM010.assign(LDGR_TRD_DTL_DVS_CD_1 = TBRLZM010.LDGR_TRD_DTL_DVS_CD.apply(lambda x : x[0:1]) )
TBRLZM010 = TBRLZM010.assign(LDGR_TRD_DTL_DVS_CD_2 = TBRLZM010.LDGR_TRD_DTL_DVS_CD.apply(lambda x : x[1:2]) )
TBRLZM010 = TBRLZM010.drop("LDGR_TRD_DVS_CD", axis = 1)

# 세부 코드 벡터라이징
value = ["C","D","M","R","T","Y"]
TBRLZM010 = dummies(TBRLZM010, "LDGR_TRD_DTL_DVS_CD_1",value)

values = ["G","D","B","C","A","N","O","L","M","H","I","T","R","S","P","Z","X"]
TBRLZM010 = dummies(TBRLZM010, "LDGR_TRD_DTL_DVS_CD_2",value)

value = ["0000","3000"]
TBRLZM010 = dummies(TBRLZM010 , "LDGR_ERR_CD" , value)

values = ["01", "02","03","04","05","06","07","08"]
TBRLZM010 = dummies(TBRLZM010, "TRLD_CARD_DVS_CD", values)

values = ["01", "02","03","04","05","06"]
TBRLZM010 = dummies(TBRLZM010, "trd_dtm_weeks", values)

# 아침 점심 저녁 벡터라이징
TBRLZM010.loc[TBRLZM010["trd_dtm_hour"] <= 24, "timedist"] = 1
TBRLZM010.loc[TBRLZM010["trd_dtm_hour"] <= 18, "timedist"] = 2
TBRLZM010.loc[TBRLZM010["trd_dtm_hour"] <= 12, "timedist"] = 3
TBRLZM010.loc[TBRLZM010["trd_dtm_hour"] <= 6, "timedist"] = 4

values = [1:4]
TBRLZM010 = dummies(TBRLZM010, "trd_dtm_hour", values)

values = [1:6]
TBRLZM010 = dummies(TBRLZM010, "trd_dtm_wom", values)

# Log transformation
TBRLZM010 = log_transform(TBRLZM010, ["BFTD_BAL","CARD_TRD_AMT", "AFT_TRD_BAL"])

# 확인
for a in TBRLZM010.columns:
	print ("value : {}".format(TBRLZM010[a].value_counts(dropna = False)))

# 형태 변환
TBRLZM010["Dir"] = TBRLZM010["Dir"].astype(int)

# nan 을 0으로
TBRLZM010.UNITRULE_CODE.replace({np.nan:0}, inplace=True)

# 음수 및 보정
data = data[(data.AFT_TRD_BAL >= 0) |
		(data.CARD_TRD_AMT >= 0) |
		(data.BFTD_BAL >= 0)]

# 차원 (현재거래 거래후 잔액 - 다음거래 거래전 잔액 차이 여부)
TBRLZM010["shift_behind_BFTD_BAL"] = TBRLZM010.sort_values(["TRCR_NO", "TRD_DTM"]).groupby("TRCR_NO")["BFTD_BAL"].shift(-1)
idx = TBRLZM010[TBRLZM010.shift_behind_BFTD_BAL.isnull()].index
TBRLZM010.loc[idx, "shift_behind_BFTD_BAL"] = TBRLZM010.loc[idx, "AFT_TRD_BAL"]
TBRLZM010 = TBRLZM010.assign(connect_AMT_DIFF_YN = (TBRLZM010.AFT_TRD_BAL != TBRLZM010.shift_behind_BFTD_BAL).astype(int,inplace = True))

# 거래 잔액 차 금액
TBRLZM010.loc[:,"AFT_minus_shift_BFTD"] = abs(TBRLZM010.loc[:, "shift_behind_BFTD_BAL"] - TBRLZM010.loc[:, "AFT_TRD_BAL"])

#null 값 확인
TBRLZM010.isnull().any()

# 데이터 내리기
filepath = os.path.join(BACKUP_PATH, 'preprocessed_data_set.pkl')

with open(filepath, 'wb') as f:
	pickle.dump(TBRLZM010, f, protocol = 4)