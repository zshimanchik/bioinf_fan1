import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler



def get_random_state():
    return 123122
    # return random.randint(0, 100000)

def get_clasterizator():
    return KMeans(n_clusters=2, random_state=get_random_state(), n_jobs=-1)
    # return DBSCAN(eps=1, n_jobs=-1)


X, y_real = make_blobs(n_samples=1500, random_state=get_random_state())
X = StandardScaler().fit_transform(X)

X2 = cdist(X, X)
# X2 = StandardScaler().fit_transform(X2)

mds = manifold.MDS(n_components=2, max_iter=300, n_init=4, random_state=get_random_state(), n_jobs=-1)
X3 = mds.fit_transform(X2)
X3 = StandardScaler().fit_transform(X3)

y_pred_orig = get_clasterizator().fit_predict(X)
y_pred_dist = get_clasterizator().fit_predict(X2)
y_pred_mds = get_clasterizator().fit_predict(X3)


plt.figure(figsize=(12, 12))

plt.subplot(231)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_orig)
plt.title("Origin")

plt.subplot(232)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_dist)
plt.title("Origin with distance clustering")

plt.subplot(233)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_mds)
plt.title("Origin with mds clustering")

plt.subplot(234)
plt.scatter(X3[:, 0], X3[:, 1], c=y_pred_orig)
plt.title("MDS with origin clustering")

plt.subplot(235)
plt.scatter(X3[:, 0], X3[:, 1], c=y_pred_dist)
plt.title("MDS with distance clustering")

plt.subplot(236)
plt.scatter(X3[:, 0], X3[:, 1], c=y_pred_mds)
plt.title("MDS with MDS clustering")

plt.show()