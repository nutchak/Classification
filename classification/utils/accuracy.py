import numpy as np


def accuracy(predicted, gold):
    return np.mean(predicted == gold)
