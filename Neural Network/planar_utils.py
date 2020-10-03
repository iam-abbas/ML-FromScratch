import numpy as np
import matplotlib.pyplot as plt

def load_data():
    
    m = 400
    N = int(m/2)
    D = 2
    X = np.zeros((m,D))
    y = np.zeros((m,1), dtype='uint8') 
    a = 4 

    # Using mathematical functions to generate petals like structure
    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2
        r = a*np.sin(4*t) + np.random.randn(N)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    X = X.T
    y = y.T
    
    return X, y

def plot_planar_data(X, y):
    
    plt.style.use('fivethirtyeight')
    plt.scatter(X[0, :], X[1, :], c=y, s=50, cmap=plt.cm.Spectral)
    plt.show()