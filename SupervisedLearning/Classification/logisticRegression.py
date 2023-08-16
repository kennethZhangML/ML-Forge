import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    predictions = sigmoid(X.dot(theta))
    cost = (-1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        gradients = (1/m) * X.T.dot(sigmoid(X.dot(theta)) - y)
        theta -= alpha * gradients
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

X = np.c_[np.ones(X.shape[0]), X]

theta = np.zeros(X.shape[1])
alpha = 0.01
num_iters = 1000

theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
print("Theta values:", theta)
