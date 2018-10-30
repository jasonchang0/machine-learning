import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
from statistics import mean

df = pd.read_csv('breast-cancer-wisconsin.data.txt')

"""
When inplace=True is passed, 
the data is renamed in place (it returns nothing)
When inplace=False is passed (this is the default value, 
so isn't necessary), 
performs the operation and returns a copy of the object
"""
df.replace('?', np.nan, inplace=True)
df.drop('id', 1, inplace=True)
df.dropna(inplace=True)

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

accuracies = []

for i in range(40):
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier(n_jobs = -1)
    clf.fit(x_train, y_train)

    accuracy = clf.score(x_test, y_test)
    print('Accuracy:', accuracy)
    accuracies.append(accuracy)

    sample_measures = np.array([[4,2,8,9,6,8,5,2,1],[2,3,1,2,3,4,3,5,1]])
    sample_measures = sample_measures.reshape(len(sample_measures), -1)

    prediciton = clf.predict(sample_measures)
    print(prediciton)

print(mean(accuracies))









