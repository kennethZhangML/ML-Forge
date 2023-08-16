import numpy as np 
from sklearn.datasets import make_classification

class DecisionTree:
    def __init__(self, depth = 0, max_depth = None):
        self.left = None 
        self.right = None 
        self.feature = None 
        self.threshold = None 
        self.label = None 
        self.depth = depth 
        self.max_depth = max_depth 
    
    def fit(self, X, y):
        num_samples, n_features = X.shape 

        unique_classes = np.unique(y)
        if len(unique_classes) == 1 or (self.max_depth and self.depth == self.max_depth):
            self.label = unique_classes[0]
            return 
        
        best_gini = float('int')
        best_split = None 
        for feature_index in range(n_features):
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                left_mask = X_column < threshold 
                right_mask = X_column >= threshold 
                gini = self.calculate_gini(y[left_mask], y[right_mask])
                if gini < best_gini:
                    best_gini = gini 
                    best_split = (feature_index, threshold)

        if best_split:
            left_mask = X[:, best_split[0]] < best_split[1]
            right_mask = X[:, best_split[0]] >= best_split[1]
            self.feature = best_split[0]
            self.threshold = best_split[1]
            self.left = DecisionTree(depth = self.depth + 1, max_depth = self.max_depth)
            self.left.fit(X[left_mask], y[left_mask])
            self.right = DecisionTree(depth = self.depth + 1, max_depth = self.max_depth)
            self.right.fit(X[right_mask], y[right_mask])
        
    def calculate_gini(self, left_labels, right_labels):
        left_size = len(left_labels)
        right_size = len(right_labels)
        total_size = left_size + right_size
        if left_size == 0 or right_size == 0:
            return 0
        
        left_prop = left_size / total_size 
        right_prop = right_size / total_size 
        left_gini = 1 - sum([(np.sum(left_labels == c) / left_size) ** 2 for c in np.unique(left_labels)])
        right_gini = 1 - sum([(np.sum(right_labels == c) / left_size) ** 2 for c in np.unique(right_labels)])
        gini = left_prop * left_gini + right_prop * right_gini 
        return gini 
    
    def predict(self, X):
        if self.label is not None:
            return self.label
        if X[self.feature] < self.threshold:
            return self.left.predict(X)
        return self.right.predict(X)

if __name__ == "__main__":
    from sklearn.datasets import make_classification 
    from sklearn.model_selection import train_test_split 

    X, y = make_classification(n_samples = 100, n_features = 4, n_redundant = 0, random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    tree = DecisionTree(max_depth = 3)
    tree.fit(X_train, y_train)

    predictions = [tree.predict(x) for x in X_test]
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    