# Networks
class Sequential:
    def __init__(self):
        self.layers = []
        self.loss = None
    
    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss):
        self.loss = loss
    
    def predict(self, X):
        self.Y = X

        for layer in self.layers:
            self.Y = layer.forward(self.Y)

        return self.Y

    def backpropagate(self, dE_dY, lr):
        grad = dE_dY

        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
    
    def fit(self, X, Y, lr, epochs=10):
        for e in range(epochs):
            Error = 0
            n = 0

            for x in X:
                pred = self.predict(x)

                target = Y[n]

                Error += self.loss.forward(pred, target)

                dE_dY = self.loss.backward(pred, target)

                self.backpropagate(dE_dY, lr)

                n += 1
            
            Error /= len(X)
            print(f"Epoch: {e}, Error: {Error}")