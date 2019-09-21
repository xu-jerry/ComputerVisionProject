# Quiz 5 - K-Nearest Neighbors

import numpy as np
from scipy import stats

class KNearestNeighbors:
    def __init__(self):
        pass
    
    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, k, X):
        #initiate the variables
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        min_indexes = np.array([])
        
        for i in range(k):
            for j in xrange(num_test):
                distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
                min_index = np.argmin(distances)
                # add the minimum indexes
                min_indexes = min_indexes.append(min_indexes, min_index)
                distances = np.remove(distances, min_index)
        
        # find the mode of the minimum indexes
        min_index = stats.mode(min_indexes)        
        Ypred[i] = self.ytr[min_index]
        return Ypred
