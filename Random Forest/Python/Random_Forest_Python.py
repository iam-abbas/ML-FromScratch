import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Salaries.csv')
print(data)

x = data.iloc[:, 1:2].values 
print(x)
y = data.iloc[:, 2].values
from sklearn.ensemble import RandomForestRegressor
  
 # create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(x, y)  
Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))
X_grid = np.arange(min(x), max(x), 0.01) 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'blue')  
  
# plot predicted data
plt.plot(X_grid, regressor.predict(X_grid), 
         color = 'green') 
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()