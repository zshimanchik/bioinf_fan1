import pickle
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

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


g = nx.from_numpy_matrix(distances)
# pos_dict = nx.random_layout(g, dim=3)
# pos_dict = nx.shell_layout(g, dim=2)  # only 2 dim supported
# pos_dict = nx.spring_layout(g, dim=3)
pos_dict = nx.fruchterman_reingold_layout(g, dim=3)
pos = np.concatenate([ list(pos_dict.values()) ])


# y_pred = DBSCAN(metric=get_distance).fit_predict([[x] for x in pos_dict.keys()])
y_pred = DBSCAN(metric="precomputed", n_jobs=-1).fit_predict(distances)
# y_pred = DBSCAN().fit_predict(pos)
# y_pred = KMeans(n_clusters=3, n_jobs=-1).fit_predict(pos)


#2D
# plt.scatter(pos[:, 0], pos[:, 1], c=y_pred)
# plt.show()

#3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=y_pred)
plt.show()