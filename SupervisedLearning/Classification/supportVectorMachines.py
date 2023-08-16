import numpy as np 
import matplotlib.pyplot as plt

def compute_costs(W, X, Y):
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0
    hinge_loss = np.sum(distances) / N
    cost = 1 / 2 * np.dot(W, W) + hinge_loss 
    return cost 

def calculate_gradient(W, X_batch, Y_batch):
    if (Y_batch * np.dot(X_batch, W)) >= 1:
        gradient = W
    else:
        gradient = W - (Y_batch * X_batch)
    return gradient 

def sgd(features, outputs):
    max_epochs = 5000 
    weights = np.zeros(features.shape[1])
    learning_rate = 0.01 
    nth = 0

    for epoch in range(1, max_epochs):
        np.random.shuffle([features, outputs])
        for ind, x in enumerate(features):
            ascent = calculate_gradient(weights, x, outputs[ind])
            weights -= (learning_rate * ascent)
    return weights 

if __name__ == "__main__":
    X = np.array([
        [2, 4],
        [4, 2],
        [1, 1],
        [3, 3]
    ])

    y = np.array([-1, -1, 1, 1])

    X = np.c_[np.ones(X.shape[0]), X]
    W = sgd(X, y)

    for d, sample in enumerate(X):
        if y[d] == -1:
            plt.scatter(sample[1], sample[2], s = 120, marker = "_", linewidths = 2)
        else:
            plt.scatter(sample[1], sample[2], s = 120, marker = "+", linewidths = 2)
    
    plt.scatter(2, 2, s = 120, marker = "_", linewidths = 2, color = 'yellow')
    plt.scatter(4, 3, s = 120, marker = "+", linewidths = 2, color = 'blue')

    x2 = [W[0] + W[1] * i for i in [0, 5]]
    plt.plot([0, 5], x2, 'k')
    plt.show()

    
