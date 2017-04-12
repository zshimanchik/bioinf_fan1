import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.distance import cdist

GSIZE = 10
PCNT = GSIZE**2
a = np.linspace(0, 1, num=GSIZE)

grid = np.meshgrid(a, a)
g = np.array(grid).reshape(2, PCNT).transpose()
# g = np.vstack([a, np.zeros(GSIZE)]).transpose()

X2 = cdist(g, g)

mds = manifold.MDS(n_components=3, max_iter=300, n_init=4, n_jobs=-1)
X3 = mds.fit_transform(X2)


# plt.subplot(211)
plt.scatter(g[:,0], g[:,1], c=range(len(g[:,0])))


# 2D
# plt.subplot(212)
# plt.scatter(X3[:,0], X3[:,1], c=range(PCNT))

# 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=range(len(g[:,0])))


plt.show()