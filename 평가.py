## K-means
from sklearn.cluster import KMeans
# k-평균으로 클러스터 추출
km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)
print("k-평균의 클러스터 크기 : {}".format(np.bincount(labels_km)))
# k-평균의 클러스터 크기 : [ 24 100  47 125  63 116 102  74   8 135]

fig, axes = plt.subplots(2, 5, figsize=(12, 4), subplot_kw={'xticks':(), 'yticks':()})  
for center, ax in zip(km.cluster_centers_, axes.ravel()):
    ax.imshow(pca.inverse_transform(center).reshape(image_shape), vmin=0, vmax=1)

import mglearn
mglearn.plots.plot_kmeans_faces(km, pca, X_pca, X_people, y_people, people.target_names)

## 병합군집
from sklearn.cluster import AgglomerativeClustering
# 병합 군집으로 클러스터 추출
agglomerative = AgglomerativeClustering(n_clusters=10)
labels_agg = agglomerative.fit_predict(X_pca)
print("k-평균의 클러스터 크기 : {}".format(np.bincount(labels_agg)))
# 병합 군집의 클러스터 크기 : [116   8 119 153  50 197 108   1   4  38]

print("ARI : {:.2f}".format(adjusted_rand_score(labels_agg, labels_km)))
# ARI : 0.06