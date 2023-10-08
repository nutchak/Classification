import numpy as np


def binary_cross_entropy(y_predicted: np.array, y: np.array) -> float:
    """A method for calculating loss (binary cross entropy)

    Parameters
    ----------
    y_predicted : `np.ndarray`
        A predicted target
    y : `np.ndarray`
        A target values

    Returns
    -------
    `float`
        Loss
    """
    return -1 / y.shape[0] * sum((y * np.log(y_predicted)) + ((1 - y) * np.log(1 - y_predicted)))
