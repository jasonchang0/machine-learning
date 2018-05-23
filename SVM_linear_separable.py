import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('dark_background')


class Support_Vector_Machine:

    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'g'}

        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data, steps=3):
        # train
        self.data = data
        # {||w||: [w,b]}
        opt_dict = {}

        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        complete_data = []

        for yi in self.data:
            for feature_set in self.data[yi]:
                for feature in feature_set:
                    complete_data.append(feature)

        self.max_feature_value = max(complete_data)
        self.min_feature_value = min(complete_data)
        complete_data = None

        step_sizes = [self.max_feature_value * 10 ** -(x + 1) for x in range(steps)]
        # point of expense: below 0.001
        # support vectors yi*(xi.w+b) ~ 1
        # ex. 1.001

        # extremely expensive
        b_range_multiple = 3
        #
        b_multiple = 5

        w_range_multiple = 2

        latest_optimum = self.max_feature_value * 10 * w_range_multiple

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # can be done because of convexity

            step_optimized = False
            while not step_optimized:
                # TODO: minimize ||w|| while maximize b
                for b in np.arange(-1 * self.max_feature_value * b_range_multiple,
                                   self.max_feature_value * b_range_multiple,
                                   step*b_multiple):
                    """
                    Since operations of b are costly and we do not need to take
                    as small of steps with b as we do with w
                    """
                    for transform in transforms:
                        w_i = w * transform
                        found_option = True
                        # Innocent until proven guilty
                        # Weakest link in the SVM fundamentally
                        # SMO attempts to fix this problem slightly

                        for y in self.data:
                            # y = 1 or -1
                            if not found_option:
                                break

                            for xi in self.data[y]:
                                invariant = y * (np.dot(w_i, xi) + b)
                                if not invariant >= 1:
                                    # This is the invariant that yi(xi.w+b) - 1 = 0
                                    found_option = False
                                    break
                                # if 1 <= invariant < 1.25:
                                #     print(xi, ':', invariant)

                        if found_option:
                            opt_dict[np.linalg.norm(w_i)] = [w_i, b]

                if w[0] < 0:
                    step_optimized = True
                    print('Step Optimized.')
                else:
                    # w = [5,5]
                    # step = 1
                    # w- [step, step]
                    w = w - step

                # print()

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            # opt_choice -> ||w|| : [w,b]

            self.w = opt_choice[0]
            self.b = opt_choice[1]

            latest_optimum = opt_choice[0][0] + step * w_range_multiple

        further_optimizable = True
        for d in self.data:
            # d = 1 or -1
            for xi in self.data[d]:
                y = d
                invariant = y*(np.dot(self.w, xi) + self.b)
                further_optimizable = invariant > 1.1 and further_optimizable
                print(xi, ':', invariant)

        if further_optimizable:
            # If one of the classes does not have a "satisfiable" results,
            # take one more step to achieve higher accuracy [expand gutter]
            self.fit(data, steps + 1)

        # pass

    def predict(self, features):
        # sign of (x.w+b)

        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=75, marker='*', c=self.colors[classification])

        return classification


    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=75, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w + b
        # v = x.w + b
        # positive support vector = 1
        # negative support vector = -1
        # decision boundaries = 0
        def hyperplane(x, w, b, v):
            # yi(xi.w + b) = 0

            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 1.2, self.max_feature_value * 1.2)
            # extra boundary on the edge of the plot

        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'w')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'w')

        # (w.x+b) = 0
        # decision boundary
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict = {-1: np.array([[1, 7], [2, 8], [3, 9], [-5, 3], [-3, 7]]),
             1: np.array([[5, 2], [6, -1], [10, 5], [-2, -5], [3, -2]])}

clf = Support_Vector_Machine()
clf.fit(data=data_dict)

to_be_predicted = [[0,10],[1,3],[3,4],[10, 15], [8, -2],[3,-4],[-5,-5],[7,-1],[-5,6],[4,7]]

for p in to_be_predicted:
    clf.predict(p)

clf.visualize()


