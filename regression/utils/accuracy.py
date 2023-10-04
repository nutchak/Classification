import numpy as np


def accuracy(predicted, goal):
    return np.mean(predicted == goal)
