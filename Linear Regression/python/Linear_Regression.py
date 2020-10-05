# Importing the dependancies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the data in numerical format
X = pd.read_csv('Linear Regression\python\Training Data\Linear_X_Train.csv').values
y = pd.read_csv('Linear Regression\python\Training Data\Linear_Y_Train.csv').values

# Standardising the data (We can even use min-max normalization)
u = X.mean()
std = X.std()
X = (X-u)/std

# Visualising the data
plt.style.use('fivethirtyeight') # Setting a plot style
plt.scatter(X, y)
plt.title("Hardwork vs Performance Graph")
plt.xlabel("Hardwork")
plt.ylabel("Performance")
plt.show()

# METHOD-1: Normal Equation

X_norm = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
theta_norm = np.dot(np.linalg.inv(np.dot(X_norm.transpose(), X_norm)), np.dot(X_norm.transpose(), y))
print(f'Normal Equation method: {theta_norm}') # Parameter values

# METHOD-2: Gradient Descent

# Hypothesis function
def cost_function(temp, learning_rate):
    cost_sigma = 0
    for i in range(X_norm.shape[0]):
        cost_sigma += (np.dot(temp.transpose(), X_norm[i])-y[i])**2
    cost = learning_rate*(1/(2*X_norm.shape[0]))*cost_sigma
    return cost

# Updating the parameters 
def update_theta(theta, learning_rate):
    temp = np.empty((theta.shape[0], 1))
    for i in range(theta.shape[0]):
        sigma = np.zeros(1, dtype=np.float64)
        for j in range(X_norm.shape[0]):
            sigma += (np.dot(theta.transpose(), X_norm[j])-y[j])*X_norm[j][i]
        temp[i] = (theta[i] - (learning_rate/X_norm.shape[0])*sum(sigma))
    return temp
    
# Iterating 1000 times (Can be reduced if the rate of change of cost is very low)
cost = [] # appending the costs for visualization
theta = np.zeros((X_norm.shape[1], 1)) # initializing the parameters
n = int(input("Enter the number of iterations: "))
for k in range(n):
    learning_rate = 0.1 # (Ideal but can be triggered)
    theta = update_theta(theta, learning_rate)
    cost.append(cost_function(theta, learning_rate))

print(f'Gradient Descent method: {theta}') 

# Visualizing the cost function
plt.plot(np.arange(n), cost)
plt.title("Cost function vs iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

# Testing the model
X_test = pd.read_csv('Linear Regression\python\Test\Linear_X_Test.csv').values # Loading data
y_test = np.dot(X_test, theta[1:].transpose()) # Computing the predictions
df = pd.DataFrame(data=y_test, columns=["y"]) # Converting to a dataframe
df.to_csv('Linear Regression\python\y_prediction.csv', index=False) # Saving the dataframe