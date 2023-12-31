import numpy as np


def normalizing(X: np.array) -> np.array:
    """A function for normalizing scaling (z-standard)

    Parameters
    ----------
    X : `np.ndarray`
        An input array

    Returns
    -------
    `np.ndarray`
        A normalizing scaled array
    """
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def min_max_scaler(X: np.array) -> np.array:
    """A function for min-maxing scaling

    Parameters
    ----------
    X : `np.ndarray`
        An input array

    Returns
    -------
    `np.ndarray`
        A min-maxing scaled array
    """
    return (X - np.min(X, axis=0)) / np.ptp(X, axis=0)