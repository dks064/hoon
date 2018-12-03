# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# 데이터 불러오기
df = pd.read_csv('../input/flavors_of_cacao.csv',na_values='\xa0')

# Nan 값 확인
b = pd.DataFrame(df.isnull().sum(), columns= ['Number of missing values'])
b

# 외도 척도 확인 
plt.figure(figsize=(8,6))
sns.distplot(df['Rating'],bins=5,color='brown')

df['Review\nDate'] = df['Review\nDate'].astype(str)

# 박스 플랏
plt.figure(figsize=(12,6))
sns.boxplot(x='Review\nDate', y='Rating',data=df)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,figsize=(12,15))

# 날짜 별 솔팅 및 확인 - 고객 묶기에도 딱좋아
a = df.groupby(['Company\nLocation'])['Rating'].mean()
a = a.sort_values(ascending=False)

b = df.groupby(['Company\nLocation'])['Rating'].median()
b = b.sort_values(ascending=False)

a = pd.DataFrame(a)
b = pd.DataFrame(b)

Ratings_by_location = a.join(b, how='left',lsuffix='_mean', rsuffix='_median')
Ratings_by_location['Mean-Median'] = Ratings_by_location['Rating_mean']-Ratings_by_location['Rating_median']
Rating_difference = sns.barplot(x=Ratings_by_location.index,y=Ratings_by_location['Mean-Median'], ax = ax3)
Rating_difference.set_xticklabels(labels = Ratings_by_location.index, rotation =90)
Rating_difference.set_ylabel("Mean-Median of ratings")


#plt.figure(figsize=(12,6))
ratings_mean = sns.barplot(x=Ratings_by_location.index,y=Ratings_by_location['Rating_mean'],ax=ax1)
ratings_mean.set_xticklabels(labels = Ratings_by_location.index, rotation =90)
ratings_mean.set_ylabel("Mean of Ratings")


#plt.figure(figsize=(12,6))
ratings_median = sns.barplot(x=Ratings_by_location.index,y=Ratings_by_location['Rating_median'], ax = ax2)
ratings_median.set_xticklabels(labels = Ratings_by_location.index, rotation =90)
ratings_median.set_ylabel("Median of ratings")

plt.tight_layout()

## 요것도 고객 묶기
plt.figure(figsize=(12,6))

c = df.groupby(['Company\nLocation'])['Cocoa % as num'].mean()
c = c.sort_values(ascending=False)

ratings = sns.barplot(x=c.index,y=c)
ratings.set_xticklabels(labels = c.index, rotation =90)

# 요것도 묶기
fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(12,15))

e = df.groupby(['Bean\nType'])['Rating'].mean()
e = e.sort_values(ascending=False)
Rating_beanType = sns.barplot(y=e.index,x=e,ax = ax1)


f = df.groupby(['Broad Bean\nOrigin'])['Rating'].mean()
f = f.sort_values(ascending=False)
Rating_broadbean = sns.barplot(y=f.index,x=f,ax = ax2)

plt.tight_layout()

# 데이터 구분하고
df1 = df[['Cocoa % as num','Rating','Review\nDate']]

# 뉴메릭 아닌 애들 빼고
#non_numerical_columns = ['Review\nDate','Bean\nType', 'Broad Bean\nOrigin','Company\nLocation']

non_numerical_columns = ['Review\nDate']

for i in non_numerical_columns:
    x1 = pd.get_dummies(df1[i])
    df1 = df1.join(x1,lsuffix='_l',rsuffix='_r')
    df1.drop(i,axis=1,inplace=True)

# 스케일러
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
df_num = StandardScaler().fit_transform(df1)

A = []
B = []
C = []

for i in np.linspace(0.1,5,50):
    db = DBSCAN(eps=i, min_samples=10).fit(df_num)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    sum = 0
    for t in labels:
        if t == -1: 
            sum = sum + 1
    C.append(sum)
            
    
    
    A.append(i)
    B.append(int(n_clusters_))

# 결과 클러스터링 확인
results = pd.DataFrame([A,B,C]).T
results.columns = ['distance','Number of clusters','Number of outliers']
results.plot(x='distance',y='Number of clusters',figsize=(10,6))

# 클러스터 적정 수
db = DBSCAN(eps=1, min_samples=10).fit(df_num)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_

# 클러스터 간 숫자(-1은 아웃라이어)
df = df.join(pd.DataFrame(labels))
df = df.rename(columns={0:'Cluster'})

df['Cluster'].value_counts()

# 그룹바이로 확인
df_clusters = df.groupby('Cluster')['Rating','Cocoa % as num']
df_clusters.describe()

# 시각화 확인
fig, (ax1,ax2) = plt.subplots(nrows = 2,figsize=(12,12))

plt.figure(figsize=(12,8))
plot1 = sns.boxplot(x=df['Cluster'],y=df['Rating'],data=df, ax = ax1)


plt.figure(figsize=(12,8))
plot2 = sns.boxplot(x=df['Cluster'],y=df['Cocoa % as num'],data=df, ax= ax2)

plt.figure(figsize=(16,12))
X = df_num

# 검은색 그래프로 노이즈 확인
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()