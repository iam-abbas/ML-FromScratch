import pandas as pd
import numpy as np
import math as math
#readingdata
train = pd.read_csv(".Dataset/train.csv")
test = pd.read_csv(".Dataset/test.csv")
#Mean and std deviation
species_group = train.groupby(['4'])
data = {
    "std_deviation": species_group.get_group('Iris-versicolor').std(),
    "mean": species_group.get_group('Iris-versicolor').mean()
}
Iris_versicolor_df = pd.DataFrame(data, columns=['mean', 'std_deviation'])


data = {
    "std_deviation": species_group.get_group('Iris-virginica').std(),
    "mean": species_group.get_group('Iris-virginica').mean()
}
Iris_virginica_df = pd.DataFrame(data, columns=['mean', 'std_deviation'])


data = {
    "std_deviation": species_group.get_group('Iris-setosa').std(),
    "mean": species_group.get_group('Iris-setosa').mean()
}
Iris_setosa_df = pd.DataFrame(data, columns=['mean', 'std_deviation'])

#calculating probability

def calculate_probability(x, mean, stdev):
    exponent = math.exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

#predict class

def get_class(feature_vector):
    Iris_setosa_prob = 0
    Iris_virginica_prob = 0
    Iris_versicolor_prob = 0
    for ind, val in enumerate(feature_vector):
        Iris_setosa_prob = Iris_setosa_prob + math.log(calculate_probability(
            val, Iris_setosa_df['mean'][ind], Iris_setosa_df['std_deviation'][ind]))
        Iris_virginica_prob = Iris_virginica_prob +math.log(calculate_probability(
            val, Iris_virginica_df['mean'][ind], Iris_virginica_df['std_deviation'][ind]))
        Iris_versicolor_prob = Iris_versicolor_prob +math.log(calculate_probability(
            val, Iris_versicolor_df['mean'][ind], Iris_versicolor_df['std_deviation'][ind]))

    minval = max([Iris_setosa_prob, Iris_virginica_prob, Iris_versicolor_prob])

    if Iris_setosa_prob == minval:
        return "Iris-setosa"
    if Iris_virginica_prob == minval:
        return "Iris-virginica"
    if Iris_versicolor_prob == minval:
        return "Iris-versicolor"

correct = 0

for row in test.iterrows():
    if get_class(row[1][:4]) == row[1][4]:
        correct = correct + 1

#predicting accuracy

print("Accuracy for Naive Bayes Classifier is", (correct/test.shape[0])*100)
