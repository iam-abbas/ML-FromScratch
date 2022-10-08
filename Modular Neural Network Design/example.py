from Layers import Dense, Activation, activations
from Losses import MeanSquaredError
from Networks import Sequential
import numpy as np

X = np.array([
    [[0, 0]],
    [[0, 1]],
    [[1, 0]],
    [[1, 1]]
])
Y = np.array([
    [[0]],
    [[1]],
    [[1]],
    [[0]]
])

nn = Sequential()
nn.add(Dense(3, inputShape=2))
nn.add(Activation(activations["tanh"]))
nn.add(Dense(5, inputShape=3))
nn.add(Activation(activations["tanh"]))
nn.add(Dense(1, inputShape=5))
nn.add(Activation(activations["sigmoid"]))

nn.compile(MeanSquaredError())

nn.fit(X, Y, 0.1, epochs=1000)

for x in X:
    print(nn.predict(x))