import pickle
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from sklearn.preprocessing.data import StandardScaler

with open('matrix.pkl', 'rb') as f:
    distances = np.array(pickle.load(f))
    # square two dimencial array wich contains distances between points
# distances = np.array([
#     [0, 5, 6],
#     [5, 0, 3],
#     [6, 3, 0],
# ])

def get_distance(i, j):
    i = int(i[0])
    j = int(j[0])
    return distances[i][j]


# X = [[x] for x in range(len(distances))]
# db = DBSCAN(eps=0.3, min_samples=10, metric=get_distance).fit_predict(X)

mds = manifold.MDS(n_components=2, n_jobs=-1)
X = mds.fit_transform(distances)
# X = StandardScaler().fit_transform(X)

y_pred = DBSCAN(n_jobs=-1).fit_predict(X)
# y_pred = KMeans(n_clusters=3).fit_predict(pos)


#2D
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

#3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred)
# plt.show()
