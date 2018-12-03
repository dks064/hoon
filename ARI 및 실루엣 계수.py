## ARI(adjusted rand index)
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np
import mglearn

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks':(), 'yticks':()})
# 3가지 알고리즘들 리스트
algos = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]
random_state = np.random.RandomState(seed=0)
random_cluster = random_state.randint(low=0, high=2, size=len(X))
# 무작위로 할당한 클러스터
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_cluster, cmap=mglearn.cm3, s=60, edgecolors='black')
axes[0].set_title("random assign - ARI : {:.2f}".format(adjusted_rand_score(y, random_cluster)))
for ax, algo in zip(axes[1:], algos):
    clusters = algo.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60, edgecolors='black')
    ax.set_title("{} - ARI: {:.2f}".format(algo.__class__.__name__, adjusted_rand_score(y, clusters)))
plt.show()
## 최적일때는 1, 무작은 0
## NMI(normalized mutual information)
## NMI를 위한 함수는 normalized_mutual_info_score로 사용법은 adjusted_rand_score와 같다.

# 타겟값 필요없는 군집용 지표 - 실루엣 계수(밀집도 정도를 확인. 높으면 좋음, 1이 최대)

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import mglearn

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

scaler = StandardScaler()
scaler.fit(X)

X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks':(), 'yticks':()})
# 3가지 알고리즘들 리스트
algos = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))
# 무작위로 할당한 클러스터
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60, edgecolors='black')
axes[0].set_title("random assign : {:.2f}".format(silhouette_score(X_scaled, random_clusters)))
for ax, algo in zip(axes[1:], algos):
    clusters = algo.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60, edgecolors='black')
    ax.set_title("{} : {:.2f}".format(algo.__class__.__name__, silhouette_score(X_scaled, clusters)))
plt.show()