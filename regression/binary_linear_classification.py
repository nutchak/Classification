import numpy as np

from regression.utils.gradient import gradient
from regression.utils.accuracy import accuracy
from regression.utils.mse import mse
from utils.add_bias import add_bias
import matplotlib.pyplot as plt


class BinaryLinearClassification:

    def __init__(self, bias=-1, threshold=0.5):
        """A class for binary linear classification

        Parameters
        ----------
        bias : `int`, default=-1
            The value of bias to add
        threshold : `float`, default=0.5
            Threshold for classification
        """
        self.bias = bias
        self.threshold = threshold
        self.weights = None
        self.losses = []
        self.accuracies = []

    def fit(self, X: np.array, y_train: {np.array, int, float}, eta: float = 0.1, epochs: int = 10):
        """A method for fitting the model (training phase)

        Parameters
        ----------
        X : `np.ndarray`
            An input array
        y_train : {`np.ndarray`, `int`, `float`}
            An array of target values
        eta : `float`, default=0.1
            A learning rate
        epochs : `int`, default=10
            A number of epochs

        Returns
        -------
        """
        # Add bias to the input array
        if self.bias:
            X_train = add_bias(X, self.bias)

        (n_samples, n_features) = X_train.shape

        # Weights is initialized as an array of zeros with (n_features) dimension
        # One for each feature plus bias
        self.weights = np.zeros(n_features)

        for e in range(epochs):
            # The predicted value in numeric for calculating loss (mse)
            y_predicted = self.numeric_predict(X)
            self.losses.append(mse(y_predicted, y_train))
            self.accuracies.append(accuracy(self.predict(X), y_train))
            # Changes in weights are minus of eta * gradient (in opposite direction)
            self.weights -= eta * gradient(X_train, y_predicted, y_train)

    def numeric_predict(self, X: np.array) -> np.array:
        """A method for calculating numerical predicted values

        Parameters
        ----------
        X : `np.ndarray`
            An input array

        Returns
        -------
        `np.array`
            An array of (numeric) predicted values
        """
        if self.bias:
            X = add_bias(X, self.bias)
        return X @ self.weights

    def predict(self, X: np.array) -> np.array:
        """A method for prediction (classification of predicted target values) from `numeric_predict()`
        The threshold is used for classification (from `__init__()`)

        Parameters
        ----------
        X : `np.array`
            An input array

        Returns
        -------
        `np.array`
            A array of classified target values
        """
        return self.numeric_predict(X) > self.threshold

    def plot_loss_accuracy(self):
        """A method for plotting losses and accuracies

        Returns
        -------
        """
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_title(label="Loss vs. epoch")
        ax1.plot(self.losses)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.set_title(label="Accuracy vs. epoch")
        ax2.plot(self.accuracies)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax1.grid()
        ax2.grid()
        fig.tight_layout()
        # fig.show()
