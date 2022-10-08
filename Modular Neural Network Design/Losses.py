import numpy as np

# Losses
class MeanSquaredError:
    def __init__(self) -> None:
        pass

    def forward(self, pred, target):
        return np.mean((target - pred) ** 2)

    def backward(self, pred, target):
        return (pred - target) * 2 / target.size
