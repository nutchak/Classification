import numpy as np


def cross_entropy(y_predicted, y):
    return -np.sum(y * np.log(y_predicted)) / y.shape[0]
