from sklearn import datasets
import pandas as pd

## 데이터 불러오고
feature = dataset[['컬럼 네임들']]
feature.head

## 모델 만들기
from sklearn.cluster import DBSCAN
import matplotlib.pyplot  as plt
import seaborn as sns
model = DBSCAN(eps=0.3,min_samples=6)
predict = pd.DataFrame(model.fit_predict(feature))
predict.columns=['predict']

## 예측
r = pd.concat([feature,predict],axis=1)
print(r)

# 플랏
sns.pairplot(r,hue='predict')
plt.show()


from mpl_toolkits.mplot3d import Axes3D
# scatter plot
fig = plt.figure( figsize=(6,6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(r['Sepal length'],r['Sepal width'],r['Petal length'],c=r['predict'],alpha=0.5)
ax.set_xlabel('Sepal lenth')
ax.set_ylabel('Sepal width')
ax.set_zlabel('Petal length')
plt.show()

## 크로스 탭으로 예측 모델 확인
ct = pd.crosstab(data['labels'],r['predict'])
print (ct)


# Standarize value
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

scaler = StandardScaler()
model = model = DBSCAN(min_samples=6)
pipeline = make_pipeline(scaler,model)
predict = pd.DataFrame(pipeline.fit_predict(feature))
predict.columns=['predict']

# concatenate labels to df as a new column
r = pd.concat([feature,predict],axis=1)

ct = pd.crosstab(data['labels'],r['predict'])
print (ct)