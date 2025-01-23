import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def finder(values, centroids):
    output = []
    color = []
    matches = []
    for i in values:
        closest = 1000000000000
        picked = [0,0]
        for n in centroids:
            check = ((i[0]-n[0])**2+(i[1]-n[1])**2)**0.5
            if check < closest:
                closest = check
                picked = n
        output.append(i)
        color.append(sum(picked))
        matches.append(picked)
    return np.array(output), np.array(color), np.array(matches)

def recentroid(values, centroids):





def firstDraft(numClusters, values):
    centroids = []
    pick = values[random.randint(0, len(values)-1)]
    numClusters -= 1
    centroids.append(pick)

    while numClusters > 0:
        farthest = 0
        new = [0,0]
        for i in values:
            bunch = []
            for n in centroids:
                check = ((i[0]-n[0])**2+(i[1]-n[1])**2)**0.5
                bunch.append(check)
            if min(bunch) > farthest:
                farthest = min(bunch)
                new = i
        centroids.append(new)
        numClusters -= 1
    return np.array(centroids)



plt.figure(figsize=(7,7))
(X, y) = make_blobs(n_samples=500, n_features=2, cluster_std=0.5, centers=10)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()


centroids = firstDraft(10, X)
(X, y, centroids) = finder(X, centroids)
recentroid(X, centroids)

fig = plt.figure(figsize=(8, 8))
plt.scatter(x=X[:,0], y=X[:,1], c = y)
plt.scatter(x=centroids[:,0], y=centroids[:,1], marker="+", c='black')
plt.show()