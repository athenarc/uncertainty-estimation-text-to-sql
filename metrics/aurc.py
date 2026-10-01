import numpy as np


class AURC:

    def __str__(self):
        return "aurc"

    def preprocess_inf(self, x, array):
        if not np.isinf(x):
            return x
        elif x > 0:
            return array.max() + 1
        else:
            return array.min() - 1

    def __call__(self, estimator: list, target: list) -> float:
        order = np.argsort(np.asarray(estimator))  # low uncertainty first
        y_sorted = np.asarray(target)[order]

        risk = 1 - np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
        return risk.mean()