import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np


style.use('dark_background')

x = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[7,10]])
x = np.append(x, [[4,5], [6,0], [1, 5], [7,2], [9,7], [5,2]], axis=0)
print(x)

# plt.scatter(x[:,0], x[:,1], s=50, color='w')
# plt.show()


colors = ['c', 'y', 'p', 'r', 'g']


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        # tol: fluctuation of centroids by percent change

        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}
        # Maintains a size of i centroids for i clusters

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            # Using new centroids when repeated in following iterations
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for feature_set in data:
                distances = [np.linalg.norm(feature_set - self.centroids[centroid])
                             for centroid in self.centroids]

                classfication = distances.index(min(distances))
                self.classifications[classfication].append(feature_set)

            prev_centroids = dict(self.centroids)
            # To avoid aliases

            for classfic in self.classifications:
                self.centroids[classfic] = np.average(self.classifications[classfic], axis=0)
                # Form new centroids by calculating the mean of features for each class

            optimized = True
            for centroid in self.centroids:
                original_centroid = prev_centroids[centroid]
                current_centroid = self.centroids[centroid]

                percent_change = np.sum((current_centroid-original_centroid)/ original_centroid*100)
                # Tolerance is calculated by percent change of centroids

                if percent_change > self.tol:
                    print('Percent Change:', percent_change)
                    optimized = False

            if optimized:
                return
        # pass

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid])
                     for centroid in self.centroids]
        print('Distances:', distances)

        classfication = distances.index(min(distances))
        return classfication
        # pass

clf = K_Means()
clf.fit(x)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker='*', color='w', s=15, linewidths=5)

for classification in clf.classifications:
    c = colors[classification]
    for feature_set in clf.classifications[classification]:
        plt.scatter(feature_set[0], feature_set[1], marker='o',
                    color=c, s=15, linewidths=5)



unknowns = np.array([[1,3], [8,9], [0, 3], [5,4], [6,4], [2,2]])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker='^', color=colors[classification],
                s=15, linewidths=5)


plt.show()


