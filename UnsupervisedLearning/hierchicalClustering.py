import numpy as np 

class HierarchicalClustering:
    def __init__(self, linkage = 'single'):
        self.linkage = linkage
        self.labels_ = None

    def fit_predict(self, X):
        N, _ = X.shape
        self.labels_ = np.arange(N)
        clusters = [[i] for i in range(N)]
        
        while len(clusters) > 1:
            c1, c2 = self._find_closest_clusters(X, clusters)
            
            clusters[c1].extend(clusters[c2])
            clusters.pop(c2)
            
            for i in clusters[c1]:
                self.labels_[i] = c1
        return self.labels_

    def _find_closest_clusters(self, X, clusters):
        min_dist = float('inf')
        c1, c2 = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = self._compute_distance(X, clusters[i], clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    c1, c2 = i, j
        
        return c1, c2

    def _compute_distance(self, X, cluster1, cluster2):
        if self.linkage == 'single':
            return np.min([np.linalg.norm(X[i] - X[j]) for i in cluster1 for j in cluster2])
        elif self.linkage == 'complete':
            return np.max([np.linalg.norm(X[i] - X[j]) for i in cluster1 for j in cluster2])
        elif self.linkage == 'average':
            return np.mean([np.linalg.norm(X[i] - X[j]) for i in cluster1 for j in cluster2])
        else:
            raise ValueError(f"Unknown linkage type: {self.linkage}")
        
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples = 50, centers = 3, random_state = 42)

    hc = HierarchicalClustering(linkage = 'single')
    labels = hc.fit_predict(X)

    print(labels)