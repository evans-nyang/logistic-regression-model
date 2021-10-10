from collections import Counter 
import numpy as np 


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # sort by distance and return indices of the first k-neighbours
        k_idx = np.argsort(distances)[: self.k]
        # extract the labels of the k-nearest neighbour training samples
        k_neighbour_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbour_labels).most_common(1)
        return most_common[0][0]



model = KNN(k=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)