import numpy as np


def add_bias(X: np.array, bias: int) -> np.array:
    """Add bias at the first column to a given input (metrix)

    Parameters
    ----------
    X : `np.ndarray`
        A numpy array to add bias to
    bias : `int`
        An integer bias to add to the numpy array
    Returns
    -------
    `np.ndarray`
        A metrix with added bias in the form [bias, X1, X2, ...]
    """
    bias_matrix = np.ones((X.shape[0], 1)) * bias
    return np.concatenate((bias_matrix, X), axis=1)
