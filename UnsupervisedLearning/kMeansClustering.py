import numpy as np 

class KMeans: 
    def __init__(self, n_clusters, max_iters = 100, tol = 1e-4):
        self.n_clusters = n_clusters 
        self.max_iters = max_iters 
        self.tol = tol 
        self.centroids = None 
    
    def fit(self, X):
        random_idx = np.random.permutation(X.shape[0])
        self.centroids = X[random_idx[:self.n_clusters]]

        for _ in range(self.max_iters):
            distances = self._compute_distances(X, self.centroids)
            labels = np.argmin(distances, axis = 1)
            prev_centroids = self.centroids.copy()

            for i in range(self.n_clusters):
                self.centroids[i] = X[labels == i].mean(axis = 0)
            
            centroid_shift = np.linalg.norm(self.centroids - prev_centroids)
            if centroid_shift < self.tol:
                break

    def predict(self, X):
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis = 1)
    
    def _compute_distances(self, X, centroids):
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis = 2)

if __name__ == "__main__":
    from sklearn.datasets import make_blobs 
    X, _ = make_blobs(n_samples = 500, centers = 3, random_state = 42)
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)

    labels = kmeans.predict(X)
    print("Centroids:\n", kmeans.centroids)
