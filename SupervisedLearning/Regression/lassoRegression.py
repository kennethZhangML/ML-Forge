import numpy as np 

def compute_cost(X, y, theta, alpha):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions - y))
    lasso_penalization = alpha * np.sum(np.abs(theta))
    return cost + lasso_penalization

def lasso_gradient_descent(X, y, theta, alpha, learning_rate, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        gradients = (1/m) * X.T.dot(X.dot(theta) - y)
        gradients_with_pen = gradients + alpha * np.sign(theta)
        theta -= learning_rate * gradients_with_pen
        J_history.append(compute_cost(X, y, theta, alpha))
    return theta, J_history 

if __name__ == "__main__":
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([2.5, 3.5, 4.5, 5.5, 6.5])

    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros(X.shape[1])
    learning_rate = 0.01 
    alpha = 0.1 
    num_iters = 1000 
    theta, cost_history = lasso_gradient_descent(X, y, theta, learning_rate, alpha, num_iters)
    print("Theta values: ", theta)

