import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import estimate_bandwidth
import random
from scipy import spatial

style.use('dark_background')

centers = random.randrange(3,7)

x, y = make_blobs(n_samples=50, centers=centers, n_features=2)
print('x:', x)
print('y:', y)

# x = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[8,2],[10,2],[9,3]])
# x = np.append(x, [[4,5], [6,0], [1, 5], [7,2], [9,7], [5,2]], axis=0)
# print(x)

# plt.scatter(x[:,0], x[:,1], s=50, color='w')
# plt.show()


colors = 10*['c', 'y', 'm', 'r', 'g']

class Mean_Shift():
    def __init__(self, radius=None, radius_norm_step=300):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        if not self.radius:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step

        centroids = []

        # All feature_set each starts as a centroid
        for i in range(len(data)):
            centroids.append(data[i])

        # weights = [i for i in range(self.radius_norm_step)][::-1]
        weights = [i**np.log(i) for i in range(self.radius_norm_step)][::-1]
        # Reverse the list of weights
        # print('weights:', weights)

        optimized = False

        print('Initial Centroids:', centroids)

        # Infinite loop
        while not optimized:
            new_centroids = []

            for i in centroids:
                in_bandwidth = []
                # centroid = centroids[i]
                centroid = i

                weight_list = []
                # To have weights for np.average

                for feature_set in data:
                    # if np.linalg.norm(feature_set - centroid) < self.radius:
                    #     in_bandwidth.append(feature_set)
                    # distance = np.linalg.norm(feature_set - centroid)
                    distance = spatial.distance.cosine(feature_set, centroid)
                    if not distance:
                        distance = 10**-9

                    print('distance:', distance)
                    print('radius:', self.radius)
                    weight_index = int(distance/self.radius)
                    print('weight_index:', weight_index)
                    if weight_index > self.radius_norm_step - 1:
                        # If the distance is greater than maximum distance,
                        # consider as lowest weight available
                        weight_index = self.radius_norm_step - 1

                    # Feature_sets are added to in_bandwidth with enhanced weights
                    # by adding weights[weight_index]**2 copies of feature_set
                    # to_be_added = [(weights[weight_index]**2)*feature for feature in feature_set]
                    # in_bandwidth.append(to_be_added)
                    # to_be_added = (weights[weight_index]**2) * [feature_set]
                    # in_bandwidth.extend(to_be_added)
                    weight_list.append(weights[weight_index])
                    in_bandwidth.append(feature_set)

                print('Weight_List:', weight_list)
                # new_centroid = np.average(in_bandwidth, axis=0)
                new_centroid = np.average(in_bandwidth, axis=0, weights=np.array(weight_list))
                try:
                    new_centroids.append(tuple(new_centroid))
                except TypeError as e:
                    pass
                # Repopulate the new_centroids set with mean of all elements that
                # are within the bandwidth of each centroid

            uniques = sorted(list(set(new_centroids)))
            # Use tuple instead of a np.array since np.unique returns unique values
            # instead of unique arrays

            to_be_popped = []
            for i in uniques:
                if i in to_be_popped:
                    continue
                for j in uniques:
                    if i == j:
                        continue
                    # elif (np.linalg.norm(np.array(i)-np.array(j)) <= self.radius
                    elif (spatial.distance.cosine(np.array(i), np.array(j)) <= self.radius
                          and j not in to_be_popped):
                        to_be_popped.append(j)
                        # break

            for _ in to_be_popped:
                uniques.remove(_)

            print('Unique:', uniques)

            prev_centroids = list(centroids)
            centroids = []
            for j in range(len(uniques)):
                centroids.append(np.array(uniques[j]))

            optimized = True

            for c in range(len(centroids)):
                if not np.array_equal(centroids[c], prev_centroids[c]):
                    optimized = False
                    break
                # if spatial.distance.cosine(centroids[c], prev_centroids[c]) > radius:
                #     optimized = False
                #     break
                # if not optimized:
                #     break
            print('Centroids:', centroids)

        self.centroids = centroids
        self.classifications = {}
        # self.classifications = []

        for i in range(len(self.centroids)):
            self.classifications[i] = []
            # self.classifications.append([])

        for feature_set in data:
            # distance = [np.linalg.norm(feature_set - self.centroids[centroid]) for centroid in self.centroids]
            distance = [spatial.distance.cosine(feature_set, centroid) for centroid in self.centroids]
            classification = distance.index(min(distance))
            self.classifications[classification].append(feature_set)


    def predict(self,data):
        # Predict method accepts data as only one feature set at a time

        # distance = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        distance = [spatial.distance.cosine(data, centroid) for centroid in self.centroids]
        # if len(distance) == 0:
        #     pass
        classification = distance.index(min(distance))
        return classification
        # pass

# radius = estimate_bandwidth(x)
radius = 0.0174533
# 1 degree in radians
print(radius)

clf = Mean_Shift(radius=radius)
clf.fit(x)

centroids = clf.centroids

plt.scatter(x[:,0], x[:,1], s=50, color='w')

for classification in clf.classifications:
    color = colors[classification]
    for feature_set in clf.classifications[classification]:
        plt.scatter(feature_set[0], feature_set[1], marker='o', color=color, s=50, linewidths=5)

for c in centroids:
    plt.scatter(c[0], c[1], s=75, color='w', marker='*')


# unknowns = np.array([[3,5], [6,7], [0, 5], [5,4], [6,10], [2,8]])
unknowns, unknown_labels = make_blobs(n_samples=20, centers=centers, n_features=2)

classifications = []
for unknown in unknowns:
    classification = clf.predict(unknown)
    classifications.append(classification)
    plt.scatter(unknown[0], unknown[1], marker='^', color=colors[classification],
                s=15, linewidths=5)

incorrect = 0
for _ in range(len(classifications)):
    if unknown_labels[_] != classifications[_]:
        incorrect += 1

print('Accuracy:', incorrect/len(classifications))

plt.show()








