import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.metrics import mean_squared_error
import seaborn as sns

unused_cols = [
	'TRCR_NO', 'UUID', 'UPD_DTM', 'RGT_DTM', 'LDGR_INTG_RFLC_DT', 'OGN_TRD_DTM'
	]

def visualize_anomaly(error_df, threshold=None):
	if threshold is None:
		threshold = error_df[error_df['Class'] == 1].reconstruction_error.quantile(q=0.95)
	print('Generated threshold : {}'.format(threshold))
	print(error_df[error_df['reconstruction_error'] > threshold].shape)
	groups = error_df.groupby('Class')
	fig , ax = plt.subplots(figsize = (20,12))
	cmap = sns.cubehelix_palette(rot=-.4, reverse = True, as_cmap = True)
	for name, group in groups:
		ax.plot(group.index, group.reconstruction_error, marker = 'o', linestyle = '', alpha = 0.6,
			label = "Fraud" if name ==1 else "Normal",
			color = 'r' if name ==1 else 'royalblue')
	ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1],
		colors = 'r', zorder = 100, label = 'Threshold')
	ax.legend

plt.title("Anomaly Visualization")
plt.xlabel("Number of Samples")
plt.ylabel("Anomaly Score")
ply.savefig("anomaly.png")

def get_anomaly_score(test, pred):
	return mean_squared_error(test, pred)