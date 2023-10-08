import numpy as np
from matplotlib import pyplot as plt

from classification.utils.accuracy import accuracy
from classification.utils.add_bias import add_bias


def cross_entropy(y_predicted, y):
    return -np.nansum(y * np.log(y_predicted)) / y.shape[0]


def binary_cross_entropy(y_predicted, y):
    return -np.sum((y * np.log(y_predicted)) + ((1 - y) * np.log(1 - y_predicted))) / y.shape[0]


def softmax(prob_logits):
    exps = np.exp(prob_logits)
    return exps / np.sum(exps, axis=1).reshape(-1, 1)


class MLPBinaryLinRegClass:
    """A multi-layer perceptron with one hidden layer"""
    def __init__(self, bias=-1, dim_hidden=6):
        """Initialize the hyperparameters"""
        self.bias = bias
        self.dim_hidden = dim_hidden

        def logistic(x):
            return 1 / (1 + np.exp(-x))

        self.activ = logistic

        def logistic_diff(y):
            return y * (1 - y)

        self.activ_diff = logistic_diff

        # Added
        self.losses = []
        self.accuracies = []
        self.losses_val = []
        self.accuracies_val = []
        self.stop_epoch = None
        self.stop_counter = 0

    def fit(self, X_train, y_train, X_val=None, y_val=None, eta=0.001, epochs=100, tol=1e-4, n_epochs_no_update=5,
            seed=None):
        """Initialize the weights. Train *epochs* many epochs.
        X_train is a Nxm matrix, N data points, m features
        y_train is a vector of length N of targets values for the training␣data,
        where the values are 0 or 1.
        """
        if X_val is not None or y_val is not None:
            assert X_val is not None and y_val is not None, "Both input data X_val and labels t_val must be provided"

        self.eta = eta

        Y_train = y_train.reshape(-1, 1)

        if X_val is not None:
            X_val_bias = add_bias(X_val, self.bias)
            Y_val = y_val.reshape(-1, 1)

        # 2 --> (1000, 2)
        dim_in = X_train.shape[1]
        # 1 --> (1000, 1)
        dim_out = Y_train.shape[1]

        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        # Initialize the weights
        self.weights1 = (rng.rand(dim_in + 1, self.dim_hidden) * 2 - 1) / np.sqrt(dim_in)
        self.weights2 = (rng.rand(self.dim_hidden + 1, dim_out) * 2 - 1) / np.sqrt(self.dim_hidden)

        X_train_bias = add_bias(X_train, self.bias)

        for e in range(epochs):

            # hidden_outs = (1000, 7) --> Outputs from hidden layer
            # outputs = (1000, 1) --> Outputs
            hidden_outs, outputs = self.forward(X_train_bias)
            t_predicted_prob = self.predict_probability(X_train)

            self.losses.append(binary_cross_entropy(t_predicted_prob, y_train))
            self.accuracies.append(accuracy(self.predict(X_train), Y_train))

            if e >= 1:
                if abs(self.losses[e - 1] - self.losses[e]) <= tol:
                    self.stop_counter += 1
                else:
                    self.stop_counter = 0

                if self.stop_counter == n_epochs_no_update:
                    self.stop_epoch = e
                    return self.stop_epoch

            # The forward step
            out_deltas = (outputs - Y_train)
            # The delta term on the output node
            hiddenout_diffs = out_deltas @ self.weights2.T
            # The delta terms at the output of the hidden layer
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * self.activ_diff(hidden_outs[:, 1:]))
            # The deltas at the input to the hidden layer
            self.weights2 -= self.eta * hidden_outs.T @ out_deltas
            self.weights1 -= self.eta * X_train_bias.T @ hiddenact_deltas
            # Update the weights

            if X_val is not None:
                t_predicted_val_prob = self.predict_probability(X_val)
                self.losses_val.append(binary_cross_entropy(t_predicted_val_prob, y_val))
                self.accuracies_val.append(accuracy(self.predict(X_val), Y_val))

    def forward(self, X):
        """Perform one forward step.
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        hidden_activations = self.activ(X @ self.weights1)
        hidden_outs = add_bias(hidden_activations, self.bias)
        outputs = self.activ(hidden_outs @ self.weights2)
        return hidden_outs, outputs

    def predict(self, X):
        """Predict the class for the members of X"""
        return self.predict_probability(X) > 0.5

    def predict_probability(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : array
            An input array

        Returns
        -------

        """
        Z = add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score = forw[:, 0]
        return score

    def plot_loss_accuracy(self, plot_validation: bool = False):
        """
        A method for plotting losses and accuracies

        Parameters
        ----------
        plot_validation : Boolean
            To plot validation set

        Returns
        -------
        """
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_title(label="Loss vs. epoch")
        ax1.plot(self.losses, label="train")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Losses")
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
        # fig.show()
