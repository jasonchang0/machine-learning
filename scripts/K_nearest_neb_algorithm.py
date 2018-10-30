import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use('dark_background')

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
            distances.append([dis, group])

    # Only select the top "K" nearest neighbors
    selections = [i[1] for i in sorted(distances)[:k]]
    print(selections)

    #[('r', 3)]
    select_result = Counter(selections).most_common(1)[0][0]

    print(Counter(selections).most_common(1))
    print(select_result)

    return select_result

dataset = {'g':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_data = [5,7]

for i in dataset:
    [plt.scatter(j[0], j[1], s=100, color=i) for j in dataset[i]]

plt.scatter(new_data[0], new_data[1], s=100, color='b')
plt.show()

result = k_nearest_neighbors(dataset, new_data, k=5)







