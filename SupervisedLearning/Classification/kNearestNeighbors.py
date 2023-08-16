import numpy as np 

from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KNN:
    def __init__(self, k = 3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X 
        self.y_train = y 
    
    def predict(self, X):
        y_pred = [self.predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common 

if __name__ == "__main__":

    data = load_iris()
    X = data.data 
    y = data.target 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    classifier = KNN(k = 3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:",accuracy * 100,"%")

