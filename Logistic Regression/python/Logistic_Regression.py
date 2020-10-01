''' We can call a Logistic Regression a Linear Regression model,
 but the Logistic Regression uses a more complex cost function,
 this cost function can be defined as the ‘Sigmoid function’ or
 also known as the ‘logistic function’ instead of a linear function.  '''

 ''' In  this program , we will perform Logistic Regression on "iris" dataset , which is available in seaborn library and try to learn the basics of Logistic Regression
 .'''



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#Load the data set
df = sns.load_dataset("iris")
df.head()


#Prepare the training set

# X = feature values, all the columns except the last column
X = df.iloc[:, :-1]

# y = target values, last column of the data frame
y = df.iloc[:, -1]


#Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=32)
model = LogisticRegression()
model.fit(x_train, y_train) #Training the model
pred = model.predict(x_test)
print(pred)# printing predictions



#Check precision, recall, f1-score
print( classification_report(y_test, pred) )

print( accuracy_score(y_test, pred))
