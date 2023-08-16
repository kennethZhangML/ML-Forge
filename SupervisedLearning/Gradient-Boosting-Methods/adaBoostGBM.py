import numpy as np 
from decisionTree import DecisionTree

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class AdaBoost:
    def __init__(self, n_estimators = 50, max_depth = 1):
        self.n_estimators = n_estimators 
        self.max_depth = max_depth 
        self.alphas = []
        self.weak_learners = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape 
        w = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            stump = DecisionTree(max_depth = self.max_depth)
            stump.fit(X, y)
            stump_pred = np.array([stump.predict(x) for x in X])

            err = np.dot(w, (stump != y).astype(np.float64))
            alpha = 0.5 * np.log((1 - err) / err)
            self.alphas.append(alpha)
            self.weak_learners.append(stump)
        
            w *= np.exp(-alpha * y * stump_pred)
            w /= np.sum(w)
    
    def predict(self, X):
        preds = np.zeros(X.shape[0])

        for alpha, learner in zip(self.alphas, self.weak_learners):
            f_preds = np.array([learner.predict(x) for x in X])
            preds += alpha * preds
        return np.sign(preds)

if __name__ == "__main__":
    X, y = make_classification(n_samples = 500, n_features = 20, random_state = 42)
    y = 2 * y - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    clf = AdaBoost(n_estimators = 50, max_depth = 1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")