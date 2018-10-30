#https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd


"""
survival: Survival (0 = No; 1 = Yes)
pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
name: Name
sex: Sex
age: Age
sibsp: Number of Siblings/Spouses Aboard
parch: Number of Parents/Children Aboard
ticket: Ticket Number
fare: Passenger Fare
cabin: Cabin
embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat: Lifeboat
body: Body Identification Number
home.dest Home/Destination
"""

style.use('dark_background')

df = pd.read_excel('titanic.xls')
# print(df.head())

df.drop(['body', 'name'], 1, inplace=True)

df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
# Example, 0 for Nan lifeboat

print(df.head())

def convert_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if not (df[column].dtype == np.int64 or df[column].dtype == np.float64):
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            # print('Unique Elements:', unique_elements)

            index = 0
            for element in unique_elements:
                if element not in text_digit_vals:
                    text_digit_vals[element] = index
                    index += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = convert_non_numerical_data(df)
# print(df.head())

df.drop(['ticket'], 1, inplace=True)

x = np.array(df.drop(['survived'], 1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

print(df.head())

clf = KMeans(n_clusters=2)
clf.fit(x)

correct = 0
for i in range(len(x)):
    """
    You have to reshape your test data because prediction 
    needs an array that looks like your training data. 
    i.e. each index needs to be a training example with 
    the same number of features as in training. 
    """
    to_be_predicted = np.array(x[i].astype(float))
    # print('Initial:', to_be_predicted)
    to_be_predicted = to_be_predicted.reshape(-1, len(to_be_predicted))
    # print('Reshaped:', to_be_predicted)
    # [to_be_predicted] -> [[to_be_predicted]]

    prediction = clf.predict(to_be_predicted)
    # print('Prediction:', prediction)

    if prediction[0] == y[i]:
        correct += 1


print('Accuracy:', correct/len(x))
# can be complemented since clusters are assigned randomly
# Such 0 -> 1 as compared to 1 -> 1 [clf.predict -> yi]








