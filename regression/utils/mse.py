import numpy as np


def mse(y_predicted: float, y: float) -> float:
    """A method for calculating mean squared error (loss)

    Formula --> 1/N * sum((y_hat - y) ** 2)

    Parameters
    ----------
    y_predicted : `float`
        A value of the predicted target (y_hat)
    y : `float`
        A value of the target (y)

    Returns
    -------
    `float`
        Mean squared error (loss)
    """
    return np.average((y_predicted - y) ** 2)
