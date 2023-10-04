import numpy as np


def sigmoid(z: float) -> float:
    """
    A sigmoid function

    Parameters
    ----------
    z : `float`
        An input

    Returns
    -------
    `float`
    """
    return 1 / (1 + np.exp(-z))
