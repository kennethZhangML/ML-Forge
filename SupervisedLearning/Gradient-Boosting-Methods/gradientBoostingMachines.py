import numpy as np 
from decisionTree import DecisionTree # import the decision tree class from before

class GradientBoostingMachine: 
    def __init__(self, n_estimators = 100, learning_rate = 0.01, max_depth = 8):
        self.n_estimators = n_estimators 
        self.learning_rate = learning_rate 
        self.max_depth = max_depth 

    def fit(self, X, y):
        y_preds = np.zeros(np.shape(y))
        for _ in range(self.n_estimators):
            residual = y - y_preds
            tree = DecisionTree(max_depth = self.max_depth)
            tree.fit(X, residual)
            self.trees.append(tree)

            y_pred += self.learning_rate * np.array([tree.predict(x) for x in X]) 
    
    def predict(self, X):
        y_pred = np.zeros((X.shape[0], ))
        for tree in self.trees:
            y_pred += self.learning_rate * np.array([tree.predict(x) for x in X])
        return y_pred 
    
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np

    X, y = make_regression(n_samples = 500, n_features = 4, noise = 0.3, random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    gbm = GradientBoostingMachine(n_estimators = 100, learning_rate = 0.1, max_depth = 3)
    gbm.fit(X_train, y_train)

    y_pred = gbm.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    r2_score = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    print(f"R-squared Score: {r2_score:.2f}")



