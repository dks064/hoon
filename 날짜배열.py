data[data.TRCR_NO.notnull() & (data.TRCR_NO ==  거시기 )
							& (data.TRD_DTM > '날짜')][["TRCR_NO"]+cols].sort_values("TRD_DTM")

data[(data.TRCR_NO == '거시기') 
		& (data.TRD_DTM > '날짜')
		& (논리)].[cols].sort_values('TRD_DTM')