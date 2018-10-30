from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('dark_background')

# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(num, var, step=2, cor=False):
    val = 1
    ys = []

    for i in range(num):
        y = val + random.randrange(-var, var)
        ys.append(y)

        if cor and cor == 'pos':
            val += step
        elif cor and cor == 'neg':
            val -= step

    xs = [i + 1 for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope(xs, ys):
    # Order of Operations: PEMDAS
    return (((mean(xs) * mean(ys)) - mean(xs*ys))
            / (mean(xs)**2 - mean(xs**2)))

def best_fit_intercept(m, xs, ys):
    return mean(ys) - m*mean(xs)

def squared_error(ys, ys_line):
    return sum((ys_line - ys)**2)

def coefficient_of_determination(ys, ys_best_fit):
    ys_mean = [mean(ys) for y in ys]
    squared_error_regr = squared_error(ys, ys_best_fit)
    squared_error_mean = squared_error(ys, ys_mean)
    return 1- (squared_error_regr/squared_error_mean)


xs, ys = create_dataset(100, 30, step = 4, cor = 'pos')
# xs, ys = create_dataset(100, 30, step = 4, cor = False)


m = best_fit_slope(xs, ys)
b = best_fit_intercept(m, xs, ys)

regression_line = [(m*x)+b for x in xs]

predict_x = 85
predict_y = (m*predict_x) + b

y_mean = [mean(ys) for y in ys]

r_squared = 1 - squared_error(ys, regression_line)/squared_error(ys, y_mean)

# print(r_squared)
print(coefficient_of_determination(ys, regression_line))


plt.scatter(predict_x, predict_y, color = 'r')
plt.scatter(xs, ys)
plt.plot(xs, regression_line)

plt.show()








