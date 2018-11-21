data = pd.read_csv("로드 .csv", parse_dates=["TRD_DTM"]

def date_parse(data, column): 
	data["TRD_DTM-year"] = data["TRD_DTM"].dt.year,
	data["TRD_DTM-month"] = data["TRD_DTM"].dt.month,
	data["TRD_DTM-day"] = data["TRD_DTM"].dt.day,
	data["TRD_DTM-hour"] = data["TRD_DTM"].dt.hour,
	data["TRD_DTM-minute"] = data["TRD_DTM"].dt.minute,
	data["TRD_DTM-second"] = data["TRD_DTM"].dt.second,

## 시각화


figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(18, 8)
sns.barplot(data=data, x="TRD_DTM-year", y="CARD_TRD_AMT", ax=ax1)
sns.barplot(data=data, x="TRD_DTM-month", y="CARD_TRD_AMT", ax=ax2)
sns.barplot(data=data, x="TRD_DTM-day", y="CARD_TRD_AMT", ax=ax3)
sns.barplot(data=data, x="TRD_DTM-hour", y="CARD_TRD_AMT", ax=ax4)
sns.barplot(data=data, x="TRD_DTM-minute", y="CARD_TRD_AMT", ax=ax5)
sns.barplot(data=data, x="TRD_DTM-second", y="CARD_TRD_AMT", ax=ax6)

figure, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
figure.set_size_inches(18, 12)
sns.pointplot(data=data, x="TRD_DTM-day", y="CARD_TRD_AMT", hue="dir",ax=ax1)
sns.pointplot(data=data, x="TRD_DTM-hour", y="CARD_TRD_AMT", hue="dir", ax=ax2)
sns.pointplot(data=data, x="TRD_DTM-second", y="CARD_TRD_AMT", hue="dir", ax=ax3)


# 전 후 열 indexing
df.getRowsBeforeLoc("리스트",3)
df.getRowsAfterLoc("리스트",3)