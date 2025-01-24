from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from random import randint

X, y = make_blobs(n_samples=500, n_features=2, cluster_std=0.7, centers=5, random_state=453)

def euclidean(point1, point2):
    """
    Euclidean distance between point & data.
    """
    return np.linalg.norm(point1-point2)

class KMeans:
    def __init__(self, n_clusters, random_state, max_iter, data):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.data = data
        self.labels = np.zeros(len(self.data))
        self.centroids = []
        self.inertia = 0
        self.iterations = 0
        np.random.seed(self.random_state)

    def fit(self):
        '''
        Creates Centroids for data (modifies self.centroids)
        '''

        #Get the first centroid
        first_centroid = self.data[randint(0, len(self.data))]
        self.centroids.append(first_centroid)

        #Get the first iteration of each of the following centroids
        while len(self.centroids) != self.n_clusters:
            distances = []

            for point in self.data:
                least_distance = float('inf')

                for centroid in self.centroids:
                    candidate_distance = euclidean(centroid, point)
                    least_distance = min(least_distance, candidate_distance)
                
                distances.append(least_distance)
                
            distances = np.array(distances)
            new_centroid = self.data[np.argmax(distances)]
            self.centroids.append(new_centroid)
            distances = []

        self.centroids = np.array(self.centroids) #Change to numpy array

        #Iterate to improve centroids
        while self.iterations < self.max_iter:
        #Assign points to the nearest centroid
            labels = self.predict()
            self.labels = labels

            #Update Centroids
            new_centroids = np.array([
                np.mean(self.data[labels == cluster], axis=0) 
                if np.any(labels == cluster) else self.centroids[cluster]
                for cluster in range(self.n_clusters) #Multi-line list comprehension is disgusting but satisfying to pull off
            ])

            self.iterations += 1

            #Convergence Check
            if np.allclose(new_centroids, self.centroids, atol=1e-6):
                break
            
            #Reassign centroids to the newly calculated set
            self.centroids = new_centroids

        return self

    def predict(self):
        '''
        Assigns points to centroid clusters, and returns an np array of those cluster assignments
        '''
        for i, point in enumerate(self.data):
            distances = [euclidean(point, centroid) for centroid in self.centroids]
            self.labels[i] = np.argmin(distances)
        return self.labels

    def fit_predict(self, data):
        '''
        Runs fit and predict functions
        '''
        self.fit()
        return self.predict(data)
    
    def score(self):
        '''
        Calculate the sum of the squared distances between each point and its centroid
        '''
        self.inertia = 0
        combined = zip(self.data, self.labels)
        #print(self.centroids)
        for point, centroid in combined:
            self.inertia += (euclidean(point, self.centroids[int(centroid)]) ** 2)
        
        return self.inertia

km = KMeans(n_clusters=5, random_state=42, max_iter=10, data=X)
km.fit()
print('fit done')
km.predict()
print('predict done')
print(km.score())

fig, ax = plt.subplots(figsize=(8, 6))
plt.title("KMeans Clustering")
plt.scatter(X[:, 0], X[:, 1], c=km.labels, cmap="tab20")
plt.scatter(km.centroids[:, 0], km.centroids[:, 1], marker='X', color='red', s=200, label="Centroids")
plt.legend()
plt.show()
