import numpy as np 
from decisionTree import DecisionTree

class RandomForest:
    def __init__(self, n_trees, max_depth = None):
        self.n_trees = n_trees 
        self.max_depth = max_depth 
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), size = len(X), replace = True)
            X_sample, y_sample = X[indices], y[indices]

            tree = DecisionTree(max_depth = self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(x) for x in X for tree in self.trees])
        tree_preds = tree_preds.reshape(X.shape[0], self.n_trees)
        return np.array([np.bincount(tree_preds[i]).argmax() for i in range(tree_preds.shape[0])])

if __name__ == "__main__":
    from sklearn.datasets import make_classification 
    from sklearn.model_selection import train_test_split 

    X, y = make_classification(n_samples = 100, n_features = 4, n_redundant = 0, random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    forest = RandomForest(n_trees = 10, max_depth = 5)
    forest.fit(X_train, y_train)
    predictions = forest.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print("Accuracy:", round((accuracy * 100), 3))