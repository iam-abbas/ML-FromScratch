import numpy as np

# Activation Functions
def relu(x, derivative=False):
    if derivative:
        return 1 if x > 0 else 0
    else:
        return x if x > 0 else 0

def tanh(x, derivative=False):
    if derivative:
        return 1 - np.tanh(x) ** 2
    else:
        return np.tanh(x)

def sigmoid(x, derivative=False):
    if derivative:
        return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
    else:
        return 1 / (1 + np.exp(-x))

activations = {
    "relu": relu,
    "tanh": tanh,
    "sigmoid": sigmoid,
    "linear": lambda x, d: x
}

# Layers
class Dense:
    def __init__(self, units, inputShape=-1, activation="linear", bias=True):
        self.inputShape = inputShape
        self.units = units
        self.weights = np.random.rand(self.inputShape, self.units)
        self.hasBias = bias
        self.bias = np.random.rand(1, self.units) if self.hasBias else np.zeroes((1, self.units))
        self.activation = Activation(activation)
        self.ouputShape = self.units

    def recomputeParameters(self):
        self.weights = np.random.rand(self.inputShape, self.units)
        self.bias = np.random.rand(1, self.units) if self.hasBias else np.zeroes((1, self.units))

    def forward(self, X):
        self.X = X
        self.Y = np.dot(X, self.weights) + self.bias

        return self.Y
    
    def backward(self, dE_dY, lr):
        # dE_dW
        self.weight_grad = np.dot(np.transpose(self.X), dE_dY)

        dE_dX = np.dot(dE_dY, np.transpose(self.weights))

        # dE_dB
        self.bias_grad = dE_dY

        self.weights -= self.weight_grad * lr
        self.bias -= self.bias_grad * lr

        return dE_dX

class Activation:
    def __init__(self, function, inputShape=-1):
        self.f = function
        self.inputShape = inputShape
        self.outputShape = self.inputShape

    def forward(self, X):
        self.X = X
        return self.f(self.X)
    
    def backward(self, dE_dY, lr):
        return self.f(self.X, derivative=True) * dE_dY

