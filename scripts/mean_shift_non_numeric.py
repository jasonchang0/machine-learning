import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
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

# style.use('dark_background')

#https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
# Keep a copy of the non-numeric data frame

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
        # if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            # print('Unique Elements:', unique_elements)

            index = 0
            for element in unique_elements:
                if element not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[element] = index
                    index += 1

            # now we map the new "id" vlaue
            # to replace the string.
            df[column] = list(map(convert_to_int, df[column]))

    return df

df = convert_non_numerical_data(df)
# print(df.head())

df.drop(['ticket', 'home.dest'], 1, inplace=True)

x = np.array(df.drop(['survived'], 1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

print(df.head())

clf = MeanShift()
clf.fit(x)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i, label in enumerate(labels):
    original_df['cluster_group'].iloc[i] = label
    # iloc provides reference to distinct rows of a column

# n_clusters_ = len(np.unique(labels))
n_clusters_ = len(cluster_centers)

survival_rates = {}
for j in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group'] == float(j)) ]
    # Conditional data frame for the original data frame
    print('Temp_df', j, ':\n', temp_df.describe())
    survival_cluster = temp_df[(temp_df['survived']==1)]

    # Avoid division by zero
    if len(temp_df) == 0:
        continue

    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[j] = survival_rate

print('Survival Rate:', survival_rates)



