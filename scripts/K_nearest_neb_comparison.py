import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


# Calculate the Euclidean distances for K-neighbor
def euclidean_distance(plot1, plot2):
    return sqrt((plot1[0] - plot2[0]) ** 2 + (plot1[1] - plot2[1]) ** 2)

def k_nearest_neighbors(data, predict, k=3):
    if len(data) > k:
        warnings.warn('K is set to a value less than total selection groups')

    distances = []
    for group in data:
        for feature in data[group]:
            # dis = euclidean_distance(feature, predict)

            dis = np.linalg.norm(np.array(feature) - np.array(predict))
            # Multi-dimensional Euclidean Distance as compared to hard-coded 2D distance formula

            distances.append([dis, group])

    # Only select the top "K" nearest neighbors
    selections = [i[1] for i in sorted(distances)[:k]]
    # print(selections)

    #[('r', 3)]
    select_result = Counter(selections).most_common(1)[0][0]
    confidence = Counter(selections).most_common(1)[0][1] / k
    # Confidence: how many points of K-neighbors belong to cohort of interest

    # print(Counter(selections).most_common(1))
    # print(select_result)

    return select_result, confidence



df = pd.read_csv('breast-cancer-wisconsin.data.txt')

df.replace('?', np.nan, inplace=True)
df.drop('id', 1, inplace=True)
df.dropna(inplace=True)
print(df)

full_data = df.astype(float).values.tolist()
print(full_data[:5])

random.shuffle(full_data)
print('\n', 30*'#', '\n')
print(full_data[:5])

accuracies = []

for i in range(40):

    test_size = 0.3
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}

    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct, total = 0, 0

    for group in test_set:
        #Group: 2 or 4
        for data in test_set[group]:
            select, confidence = k_nearest_neighbors(train_set, data, 5)
            if (group == select):
                correct += 1
            else:
                print(confidence)
            total += 1

    print('Accuracy:', correct/total)
    accuracies.append(correct/total)


print(sum(accuracies)/len(accuracies))









