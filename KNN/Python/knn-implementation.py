#k-nearest neighbours implementation
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

dfx = pd.read_csv("./ydata")
dfy = pd.read_csv("./xdata")

x = dfx.values
y = dfy.values
#we drop the first coloumn
x= x[:,1:]
y= y[:,1:].reshape(-1,)

#here we are ploting the graph
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()

#let's generate one point 
query_x = np,array([2,3])
plt.scatter(x[:,0],x[:,1],c=y)
plt.scatter(query_x[0],query_x[1],color = 'red')
plt.show()

#now we have to create distance matrix

def dis(x1,x2):
	return np.sqrt(sum(x1-x2)**2)

def knn(x,y,querypoint,k=5):
	vals = []
	n= x.shape()
	for i in range(m):
		d = dist(querypoint,x[i])
		vals.append((d,y[i]))

	vals = sorted(vals)
	#nearest first k points 
	vals = vals[:k]
	vals = np.array(vals)
	#printing the vals
	print(vals)
	#this line of code generate the count of unique numbers
	new_vals = np.unique(vals[:,1],return_counts=True)
	index = new_vals[1].argmax()
	pred = new_vals[0][index]
	return pred 

x = knn(x,y,query_x)
print(x)












