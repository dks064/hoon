import numpy as np
import pandas as pd
from math import ceil

RANDOM_STATE = 42
LOAD_PATH = "/sorc001/gdssvc01/notebook/data"
BACKUP_PATH = "/sorc001/gdssvc01/notebook/backupdata/sup_model_1"
REULT_PATH = "/sorc001/gdssvc01/notebook/result/sup_model_1"

TB010_dtypes = {"UUID": str,
"TRCR_NO" : str,

"TRD_DTM" : str,
"LDGR_TRD_DTL_DVS_CD" : str,
"CLOS_DT" : str,
"LDGR_TBDS_DVS_CD" : str,
"STN_ID" : str,
"TRNS_TRD_TRNC_ID" : str,
"FCTT" : str,
"FRC_IT" : str,
"BFTD_DTM" : str,
"BFTD_BAL" : str,
"CARD_TRD_SNO" : str,
"LDGR_TRD_DVS_CD" : str,
"SMCR_MDL_ID" : str,
"TRCR_AFCP_ID" : str,
"CGR_ID" : str,
"LDGR_ERR_CD" : str,
"OGN_TRD_UPD_YN" : str,
"OGN_TRD_DTM" : str,
"STST_EX_RSN_CD" : str,
"SAM_ID" : str,
"TRLD_CARD_DVS_CD" : str,
"STAF_PRD_ID" : str,
"RGSR_ID" : str,
"RGT_DTM" : str,
"UPDR_ID" : str,
"UPD_DTM" : str,
"ANLY_FRC_ID" : str,}
gf_fds_dtypes = {"UUID" : str,
"TARGET_UUID" : str,
"TARGET_TABLE" : str,
"TARGET_LOG_DTTM" : str,
"TARGET_ID" : str,
"RULE_TYPE" : str,
"RULE_TARGET" : str,
"SERVICE_TYPE" : str,
"RULESET_CODE" : str,
"OFFICIAL_CODE" : str,
"FINNAL_HANDLING" : str,
"TRANS_DTTM" : str,
"MEASURE_DTTM" : str,
"REG_STAMP" : str,
"UNITRULE_CODE" : str,
"SM_AMOUNT" : np.float32,
"SM_CO" : np.float32,
"DETCT_TY_CODE" : str, 
"MODEL_ID" : str, 
"SCORE" : np.float32,
"ACDNT_YN" : str}

def date_parser(strings , format_str):
	return pd.datetimne.strptime(strings, format_str)

def dummies(data , column, lists):
	for kcol in lists:
		colname = str(column) + "_" + str(col)
		data[colname] = (data[column] == col).astype(int)
	data = data.drop(column , axis = 1)
	return data

def log_transform(data, columns):
	df_log = data[columns].apply(lambda x :np.log10(x + 1))
	df_log.columns = 'log_' + df_log.columns
	data = pd.concat([data, df_log], axis = 1)
	return data

def save_plk(save_filename, objects, modes):
	import pickle
	with open(save_filename, modes) as f:
		pickle.dump(objects,f)

def week_of_modth(dt)
	""" return wwek of month specified day
	(specified day's month's weekday(monday = 0) + specified day) / 7 ceiling
	Arguments:
		(datetime) -- specified day
	returns:
		(int) -- week of month
	"""
	first_day = dt.replace(day = 1)
	adjusted_day = dt.day + first_day.weekday()
	return int(ceil(adjusted_day/7))

