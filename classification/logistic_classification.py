import numpy as np
from matplotlib import pyplot as plt

from classification.utils.accuracy import accuracy
from classification.utils.add_bias import add_bias
from classification.utils.binary_cross_entropy import binary_cross_entropy
from classification.utils.gradient import gradient
from classification.utils.sigmoid import sigmoid


class LogisticClassification:
    def __init__(self, bias=-1, threshold=0.5):
        """A class for logistic classification

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
        self.losses_val = []
        self.accuracies_val = []
        self.stop_epoch = None

    def predict_probability(self, X: {np.array, float}) -> {np.array, float}:
        """A method for predicting probability

        Parameters
        ----------
        X : {`np.ndarray`, `float`}
            An input array

        Returns
        -------
        {`np.ndarray`, `float`}
            An array of probabilities
        """
        if self.bias:
            z = add_bias(X, self.bias)
        return sigmoid(z @ self.weights)

    def predict(self, X: np.array) -> np.array:
        """A method for prediction

        Parameters
        ----------
        X : `np.ndarray`
            An input array

        Returns
        -------
        `np.ndarray`
            Classification of the predicted values (an array of 0 (negative class) and 1 (positive class))
        """
        return (self.predict_probability(X) > self.threshold).astype("int")

    def fit(self, X: np.array, y_train: np.array, X_val: np.array = None, y_val: np.array = None, eta: float = 0.1, epochs: int = 10, tol: float = 1e-5, n_epochs_no_update: int = 5):
        """A method for fitting the model (training phase)

        Parameters
        ----------
        X : `np.ndarray`
            An input array
        y_train : `np.ndarray`
            An array of target values
        X_val : `np.ndarray`, default=None
            An input array of validation set (initialized as None)
        y_val : `np.ndarray`, default=None
            An array of target values (initialized as None)
        eta : `float`, default=0.1
            A learning rate
        epochs : `int`, default=10
            A number of epochs
        tol : `float`, default=1e-5
            Tolerance
        n_epochs_no_update : `int`, default=5
            A counter of number of epochs

        Returns
        -------
        """
        # Check if the X_val and t_val are given
        if X_val is not None or y_val is not None:
            assert X_val is not None and y_val is not None, "Both input data X_val and labels t_val must be provided"

        # A counter for counting epochs
        self.stop_counter = 0

        if self.bias:
            X_train = add_bias(X, bias=self.bias)

        (n_samples, n_features) = X_train.shape

        self.weights = np.zeros(n_features)

        for e in range(epochs):

            t_predicted_prob = self.predict_probability(X)
            self.losses.append(binary_cross_entropy(t_predicted_prob, y_train))

            if e >= 1:
                # Difference of losses between current and previous epochs
                if abs(self.losses[e-1] - self.losses[e]) <= tol:
                    self.stop_counter += 1
                else:
                    self.stop_counter = 0

                if self.stop_counter == n_epochs_no_update:
                    self.stop_epoch = e
                    return self.stop_epoch

            self.accuracies.append(accuracy(self.predict(X), y_train))
            self.weights -= eta * gradient(X_train, t_predicted_prob, y_train)

            if X_val is not None:
                t_predicted_val_prob = self.predict_probability(X_val)
                self.losses_val.append(binary_cross_entropy(t_predicted_val_prob, y_val))
                self.accuracies_val.append(accuracy(self.predict(X_val), y_val))

    def plot_loss_accuracy(self, plot_validation: bool = False):
        """A method for plotting losses and accuracies

        Parameters
        ----------
        plot_validation : bool, default=False
            To plot validation set

        Returns
        -------
        """
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_title(label="Loss vs. epoch")
        ax1.plot(self.losses, label="train")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.set_title(label="Accuracy vs. epoch")
        ax2.plot(self.accuracies, label="train")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax1.grid()
        ax2.grid()
        fig.tight_layout()
        if plot_validation:
            ax1.plot(self.losses_val, label="val", alpha=0.5)
            ax2.plot(self.accuracies_val, label="val", alpha=0.5)
            ax1.legend()
            ax2.legend()
        #fig.show()
