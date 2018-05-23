from unittest import TestCase
from linear_regr_algorithm import create_dataset, best_fit_slope
from linear_regr_algorithm import best_fit_intercept, coefficient_of_determination


class test_is_best_fit(TestCase):
    def test_fit(self):


        xs, ys = create_dataset(100, 30, step=4, cor='pos')

        m = best_fit_slope(xs, ys)
        b = best_fit_intercept(m, xs, ys)

        regression_line = [(m * x) + b for x in xs]

        # self.assertTrue('FOO'.isupper())
        self.assertTrue(coefficient_of_determination(ys, regression_line) > 0.95)
