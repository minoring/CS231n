import numpy as np


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """
          X is N x D (50000 * 3072) where each row is an example.
          Y is 1-dimension of size N
        """
        # The nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k):
        """
          X is N x D where each row is an example we wish to predict label for
        """
        num_test = X.shape[0]

        # Lets make sure that the output type matches the input typee
        Ypred = np.zeros((num_test, k), dtype=self.ytr.dtype)

        # Loop over all test rows
        for i in range(num_test):
            print(f'Predictiong {i}th element')
            # Find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_indices = np.argpartition(distances, k)[:k]
            Ypred[i] = self.ytr[min_indices]

        return Ypred
