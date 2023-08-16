import numpy as np 
from scipy.stats import multivariate_normal 

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split 

class GMM:
    def __init__(self, n_components = 3, max_iters = 1000, tol = 1e-4):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None

    def fit(self, X):
        n_samples, n_features = X.shape 
        self.weights_ = np.ones(self.n_components) / self.n_components
        random_idx = np.random.choice(n_samples, self.n_components, replace = False)
        self.means_ = X[random_idx]
        self.covariances_ = [np.eye(n_features) for _ in range(self.n_components)]

        prev_ll = float("-inf")
        for _ in range(self.max_iters):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            l1 = self._compute_log_likelihood(X)

            if np.abs(l1 - prev_ll) < self.tol:
                break
            prev_ll = l1
    
    def _e_step(self, X):
        probs = np.array([self.weights_[j] * multivariate_normal(self.means_[j], self.covariances_[j]).pdf(X) 
                                    for j in range(self.n_components)])
        responsibilities = probs / probs.sum(axis=0)
        return responsibilities

    def _m_step(self, X, responsibilities):
        for j in range(self.n_components):
            weight_j = responsibilities[j].sum()
            mean_j = (responsibilities[j][:, np.newaxis] * X).sum(axis = 0) / weight_j
            covariance_j = np.dot((responsibilities[j][:, np.newaxis] * (X - mean_j)).T, (X - mean_j)) / weight_j

            self.weights_[j] = weight_j / len(X)
            self.means_[j] = mean_j
            self.covariances_[j] = covariance_j

    def _compute_log_likelihood(self, X):
        return np.sum(np.log(np.array([self.weights_[j] * multivariate_normal(self.means_[j], self.covariances_[j]).pdf(X) 
                                       for j in range(self.n_components)]).sum(axis = 0)))

    def predict(self, X):
        return np.argmax(self._e_step(X), axis = 0)
    
if __name__ == "__main__":

    X, y = make_blobs(n_samples = 300, centers = 3, cluster_std = 0.60, random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    gmm = GMM(n_components = 3, max_iters = 100, tol = 1e-4)
    gmm.fit(X)
