import numpy as np


def gradient(X: np.array, y_predicted: {np.array, float}, y: {np.array, float}) -> float:
    """A method for calculating a gradient

    Parameters
    ----------
    X : `numpy.ndarray`
        An input array
    y_predicted : {`np.ndarray`, `float`}
        A value of the predicted target (y_hat)
    y : {`np.ndarray`, `float`}
        A value of the target (y)

    Returns
    -------
    `float`
        A gradient
    """
    return X.T @ (y_predicted - y) / X.shape[0]

"""
`int`

`float`

`np.ndarray`

{np.array, float}

{`np.ndarray`, `float`}

"""
