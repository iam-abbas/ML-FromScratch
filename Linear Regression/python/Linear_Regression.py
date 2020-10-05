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
print("\nComputing parameters using Normal Equation ...\n")
X_norm = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
theta_norm = np.dot(np.linalg.inv(np.dot(X_norm.transpose(), X_norm)), np.dot(X_norm.transpose(), y))
print(f'Normal Equation method params: {theta_norm}')  # Parameter values

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
    
print("\nComputing parameters using Gradient Descent...\n")
# Iterating 1000 times (Can be reduced if the rate of change of cost is very low)
cost = [] # appending the costs for visualization
theta = np.zeros((X_norm.shape[1], 1)) # initializing the parameters
n = int(input("Enter the number of iterations: "))
for k in range(n):
    learning_rate = 0.1 # (Ideal but can be triggered)
    theta = update_theta(theta, learning_rate)
    cost.append(cost_function(theta, learning_rate))

print(f'Gradient Descent method params: {theta}') # Parameter values

# METHOD-3: Gradient Descent with Regularization

# Hypothesis function
def cost_function_with_rg(temp, learning_rate, lmb):
    cost_sigma = 0
    for i in range(X_norm.shape[0]):
        cost_sigma += ((np.dot(temp.transpose(), X_norm[i])-y[i])**2) + (lmb*(np.sum(np.square(temp))))
    cost = learning_rate*(1/(2*X_norm.shape[0]))*cost_sigma
    return cost

# Updating the parameters 
def update_theta_with_rg(theta, learning_rate, lmb):
    temp = np.empty((theta.shape[0], 1))
    for i in range(theta.shape[0]):
        sigma = np.zeros(1, dtype=np.float64)
        for j in range(X_norm.shape[0]):
            sigma += ((np.dot(theta.transpose(), X_norm[j])-y[j])*X_norm[j][i]) + (lmb)*theta[i][0]
        temp[i] = (theta[i] - (learning_rate/X_norm.shape[0])*sum(sigma))
    return temp
    
print("\nComputing parameters using Gradient Descent with regularization...\n")
# Iterating 1000 times (Can be reduced if the rate of change of cost is very low)
cost_rg = [] # appending the costs for visualization
theta_rg = np.zeros((X_norm.shape[1], 1)) # initializing the parameters
nreg = int(input("Enter the number of iterations: "))
lmb = float(input("Enter the lambda to be used for regularization (usually 0.01): "))
for k in range(nreg):
    learning_rate = 0.1 # (Ideal but can be triggered)
    theta_rg = update_theta_with_rg(theta_rg, learning_rate, lmb)
    cost_rg.append(cost_function_with_rg(theta_rg, learning_rate, lmb))

print(f'Gradient Descent method with regularization params: {theta_rg}') # Parameter values



# Visualizing the cost function
plt.plot(np.arange(n), cost)
plt.plot(np.arange(nreg), cost_rg)
plt.title("Cost function vs iterations")
plt.legend(['Without reg','With reg'])
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

print("\nCalculating for test ... \n")
th = int(input("Enter your choice of param (1 for with regularization 0 for without : "))
# Testing the model
X_test = pd.read_csv('Linear Regression\python\Test\Linear_X_Test.csv').values # Loading data
if th:
    y_test = np.dot(X_test, theta_rg[1:].transpose()) # Computing the predictions
else:
    y_test = np.dot(X_test, theta_rg[1:].transpose()) # Computing the predictions
df = pd.DataFrame(data=y_test, columns=["y"]) # Converting to a dataframe
print(f"\n Prepared test result dataframe of shape {df.shape}")
df.to_csv('Linear Regression\python\y_prediction.csv', index=False) # Saving the dataframe