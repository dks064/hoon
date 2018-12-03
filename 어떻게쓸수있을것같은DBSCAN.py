from sklearn.decomposition import PCA
import numpy as np

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# 0~255 사이의 흑백 이미지를 픽셀 값을 0~1 스케일로 조정
# MinMaxScaler 적용과 비슷
X_people = X_people / 255.

pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit_transform(X_people)
X_pca = pca.transform(X_people)

from sklearn.cluster import DBSCAN
dbscan = DBSCAN()
labels = dbscan.fit_predict(X_pca)
print("고유한 레이블 \n{}".format(np.unique(labels)))

# min-sample = 3, eps = 15
dbscan = DBSCAN(min_samples=3, eps=15)
labels = dbscan.fit_predict(X_pca)
print("고유한 레이블 \n{}".format(np.unique(labels)))

# 잡음 포인트와 클러스터에 속한 포인트 수 세기
# bincount는 음수를 받을수 없어 +1을 한다.
print("클러스터 별 포인트 수: {}".format(np.bincount(labels + 1)))

# 잡음 포인트와 클러스터에 속한 포인트 수 세기

# bincount는 음수를 받을수 없어 +1을 한다.

print("클러스터 별 포인트 수: {}".format(np.bincount(labels + 1)))
noise = X_people[labels == -1]
fig, axes = plt.subplots(3, 9, subplot_kw={'xticks':(), 'yticks':()}, figsize=(12, 4))
for image, ax in zip(noise, axes.ravel()):
    ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)

# 더 많은 클러스터를 찾으려면 eps를 0.5 ~ 15 사이 정도로
# 잡음 포인트와 클러스터에 속한 포인트 수 세기
# bincount는 음수를 받을수 없어 +1을 한다.
for eps in [1, 3, 5, 7, 9, 11, 13]:
    print("\neps={}".format(eps))
    dbscan = DBSCAN(eps=eps, min_samples=3)
    labels = dbscan.fit_predict(X_pca)    
    print("클러스터 수 {}".format(len(np.unique(labels))))
    print("클러스터 크기 {}".format(np.bincount(labels + 1)))

dbscan = DBSCAN(min_samples=3,eps=7)
labels = dbscan.fit_predict(X_pca)
for cluster in range(max(labels) + 1):
    mask = labels == cluster
    n_images = np.sum(mask)
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4), subplot_kw={'xticks':(), 'yticks':()})  
    for image, label, ax in zip(X_people[mask], y_people[mask], axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        ax.set_title(people.target_names[label].split()[-1])

