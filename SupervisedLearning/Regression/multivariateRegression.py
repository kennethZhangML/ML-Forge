import numpy as np 

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    # This can be defined differently
    cost = (1/2 * m) * np.sum(np.square(predictions - y))
    return cost 

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        grads = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta -= alpha * grads 
        J_history.append(compute_cost(X, y, theta))
        return theta, J_history

def feature_normalize(X):
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    X_norm = (X - mean) / std 
    return X_norm, mean, std 

def main():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([2.5, 3.5, 4.5, 5.5, 6.5])

    X, mean, std = feature_normalize(X)
    X = np.c_[np.ones(X.shape[0]), X]

    theta = np.zeros(X.shape[1])
    alpha = 0.01 
    num_iters = 1000 

    theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
    print("Theta Values: ", theta)

if __name__ == "__main__":
    main()