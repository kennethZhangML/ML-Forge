import numpy as np 

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2 * m) * np.sum(np.square(predictions - y)) # mean squared error cost function 
    return cost 

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters): # number of iterations 
        grads = (1/m) * X.T.dot(X.dot(theta) - y) # compute cost function  
        theta -= alpha * grads # compute theta via update rule 
        J_history.append(compute_cost(X, y, theta)) 
    return theta, J_history 

def main():
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2.5, 3, 4.5, 5])

    X = np.c_[np.ones(X.shape[0]), X]

    theta = np.zeros(2)
    alpha = 0.001
    num_iters = 1000
    theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
    print("Theat Values: ", theta)

if __name__ == "__main__":
    main()
