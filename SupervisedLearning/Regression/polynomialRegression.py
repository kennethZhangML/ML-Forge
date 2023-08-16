import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        gradients = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= alpha * gradients
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

def feature_normalize(X):
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    X_norm = (X - mean) / std
    return X_norm, mean, std

def polynomial_features(X, degree):
    X_poly = X
    for i in range(2, degree + 1):
        X_poly = np.c_[X_poly, np.power(X, i)]
    return X_poly

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

degree = 2
X_poly = polynomial_features(X, degree)

X_poly, mean, std = feature_normalize(X_poly)

X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]

theta = np.zeros(X_poly.shape[1])
alpha = 0.01
num_iters = 1000

theta, J_history = gradient_descent(X_poly, y, theta, alpha, num_iters)
print("Theta values:", theta)
