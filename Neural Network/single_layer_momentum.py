# Importing the dependancies
import numpy as np
import matplotlib.pyplot as plt
from planar_utils import *
# Loading the data 
X, y = load_data()

# Visualizing the data
plot_planar_data(X, y)

# Defining the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Defining the relu function
def relu(z):
    return z * (z > 0)

# Defining the derivative during back propagation
def relu_derivative(z):
    return 1 * (z > 0)

# Defining some global variables
hidden_neurons = 10 
epochs = 10000
m = X.shape[1]
learning_rate = 1.2
cost = [0]*epochs # For visualizaion
epsilon = 1e-8

# Initializing the weights
W1 = np.random.randn(hidden_neurons, X.shape[0])*0.01
b1 = np.zeros((hidden_neurons, 1)) 
W2 = np.random.randn(y.shape[0], hidden_neurons)*0.01
b2 = np.zeros((y.shape[0], 1))


v_t = np.zeros((4,))

for epoch in range(epochs):
    
    # Forward propagation
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    # Cost 
    logprobs = np.multiply(np.log(A2 + epsilon), y) + np.multiply((1 - y), np.log(1 - A2 + epsilon))

    cost[epoch] = - np.sum(logprobs) / m
    if epoch%1000 == 0: # Printing the cost after every 1000 iterations for debugging
        print(f'Cost after {epoch} iterations = {cost[epoch]}') 
    
    # Backward propagation
    dZ2= A2 - y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = np.array([dW2, db2, dW1, db1])
    # caculating the exponential average.
    v_t = 0.8*v_t + learning_rate*grads
    # Updating the parameters
    W2 -= v_t[0]
    b2 -= v_t[1]
    W1 -= v_t[2]
    b1 -= v_t[3]

plt.style.use('fivethirtyeight')
        
# Visualizing the cost function
plt.plot(cost, label = 'cost')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.legend()
plt.show()

# Predicting 
Z1 = np.dot(W1, X) + b1
A1 = np.tanh(Z1)

Z2 = np.dot(W2, A1) + b2
y_pred = sigmoid(Z2)

print ('Accuracy: %d' % float((np.dot(y, y_pred.T) + np.dot(1-y, 1-y_pred.T))/float(y.size)*100) + '%')