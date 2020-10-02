# Importing the dependancies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode

# Loading the data in numerical format
X = pd.read_csv("KNN/Python/xdata.csv").values
y = pd.read_csv("KNN/Python/ydata.csv").values
X = X[:, 1:]

# Splitting the dataset 
train_size = int(0.8*len(y)) # Using the last 20% as test set. Shuffle the dataset before splitting (recommended)

X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Creating a class
class kNN():
    
	# Constructor with k, distance function as inputs
    def __init__(self, k=1, distance='L1'):
        self.k = k
        self.distance = distance
        
	# Training the model includes storing the data for comparison during testing
    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
	# Testing 
    def predict(self, X_test):
        
		# Initializing the predicted array
        y_pred = np.zeros(X_test.shape[0], dtype = self.y_train.dtype)
        
		# L1 norm (difference between values)
        if self.distance=='L1':
            for i in range(X_test.shape[0]):
                diff = np.sum(np.abs(X_test[i, :], self.X_train), axis=1)
                y_pred[i] = mode(self.y_train[a[np.argpartition(a, k)[:k]]]) # Returning the mode of the nearest classes
                
		# L2 norm (square root of the difference of sum of squares)
        elif self.distance=='L2':
            for i in range(X_test.shape[0]):
                diff = np.sqrt(np.sum(np.square(X_test[i, :], self.X_train), axis=1))
                y_pred[i] = mode(self.y_train[a[np.argpartition(a, k)[:k]]])
            
        else:
            raise Exception("Wrong distance input")
            
        return y_pred

knn_model = kNN(k=5)
knn_model.train(X_train, y_train)
pred = knn_model.predict(X_test)
print(f'Accuracy = {np.mean(y_test==pred)}')