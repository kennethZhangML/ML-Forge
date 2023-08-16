import numpy as np 
from decisionTree import DecisionTree 

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class XGBoostLikeGBM: 
    def __init__(self, n_estimators = 100, learning_rate = 0.01, max_depth = 8, reg_lambda = 1.0):
        self.n_estimators = n_estimators 
        self.learning_rate = learning_rate 
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda  # Regularization term
        self.trees = []

    def fit(self, X, y):
        y_preds = np.zeros(np.shape(y))
        for _ in range(self.n_estimators):
            residual = y - y_preds + self.reg_lambda * y_preds

            tree = DecisionTree(max_depth = self.max_depth)
            tree.fit(X, residual)
            self.trees.append(tree)

            y_preds += self.learning_rate * np.array([tree.predict(x) for x in X]) 

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], ))
        for tree in self.trees:
            y_pred += self.learning_rate * np.array([tree.predict(x) for x in X])
        return y_pred 

if __name__ == "__main__":

    X, y = make_classification(n_samples = 500, n_features = 20, random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    model = XGBoostLikeGBM(n_estimators = 100, learning_rate = 0.1, max_depth = 3, reg_lambda = 1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    r2_score = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    print(f"R-squared Score: {r2_score:.2f}")