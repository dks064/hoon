# 이상치 처리
outlier = data[(data["BFTD_BAL"] < "0") | (data["BFTD_BAL"] > "50000")]
outlier = data[(data["CARD_TRD_AMT"] < "0") | (data["CARD_TRD_AMT"] > "50000")]
outlier = data[(data["AFT_TRD_BAL"] < "0") | (data["AFT_TRD_BAL"] > "50000")]

outlier = data[(data["AFT_TRD_BAL"] < "0") & (data["AFT_TRD_BAL"] > "50000")]
outlier.desc()