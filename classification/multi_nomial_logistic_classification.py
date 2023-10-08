import numpy as np
from matplotlib import pyplot as plt

from classification.utils.accuracy import accuracy
from classification.utils.add_bias import add_bias
from classification.utils.classifier import classifier
from classification.utils.cross_entropy import cross_entropy
from classification.utils.gradient import gradient
from classification.utils.one_hot_encoding import one_hot_encoding
from classification.utils.softmax import softmax


class MultinomialClassification:
    def __init__(self, bias=-1):
        self.bias = bias
        self.weights = None
        self.losses = []
        self.accuracies = []

    # Change to predict
    def forward(self, X):
        if self.bias:
            z = add_bias(X, self.bias)
        return softmax(z @ self.weights)

    def fit(self, X, y_train, eta=0.1, epochs=10):

        if self.bias:
            X_train = add_bias(X, bias=self.bias)

        # (1000, 3)
        (n_samples, n_features) = X_train.shape

        # 5
        n_classes = np.unique(y_train).shape[0]
        # (1000, 5)
        y_ova = one_hot_encoding(y_train, n_classes)

        self.weights = np.zeros((n_features, n_classes))

        for e in range(epochs):
            # (1000, 5)
            y_predicted = self.forward(X)
            self.losses.append(cross_entropy(y_predicted, y_ova))
            self.accuracies.append(accuracy(self.predict(X), y_train))
            self.weights -= eta * gradient(X_train, y_predicted, y_ova)

    def predict(self, X):
        if self.bias:
            z = add_bias(X, self.bias)
        return classifier(softmax(z @ self.weights))

    def plot_loss_accuracy(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_title(label="Loss vs. epoch")
        ax1.plot(self.losses)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Losses")
        ax2.set_title(label="Accuracy vs. epoch")
        ax2.plot(self.accuracies)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax1.grid()
        ax2.grid()
        fig.tight_layout()
        # fig.show()
