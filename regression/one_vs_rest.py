import numpy as np

from regression.logistic_classification import LogisticClassification


class OneVsRestLogisticRegression:
    def __init__(self):
        """A class for multinomial logistic classification (one-vs-rest, inherit from `NumpyLogReg()`)

        Parameters
        ----------
        """
        super().__init__()
        self.classifiers = None

    def fit(self, X: np.array, y_train: np.array, X_val: np.array = None, y_val: np.array = None, eta: float = 0.1, epochs: int = 10, tol: float = 1e-4, n_epochs_no_update: int = 5):
        """A method for fitting the model (training phase)

        Parameters
        ----------
        X : np.array
            An input array
        y_train : np. array
            An array of target values
        X_val : array, default=None
            An input array of validation set (initialized as None)
        y_val : array, default=None
            An array of target values (initialized as None)
        eta : float, default=0.1
            A learning rate
        epochs : int, default=10
            A number of epochs
        tol : float, default=1e-4
            Tolerance
        n_epochs_no_update : int, default=5
            A counter of number of epochs

        Returns
        -------
        """
        n_classes = len(np.unique(t_multi_train))
        # Create classifier for each class
        self.classifiers = [LogisticClassification() for _ in range(n_classes)]

        for i, classifier in enumerate(self.classifiers):
            # Change the target values to 1 if equals to i otherwise 0
            t_train_bi = np.where(y_train == i, 1, 0)
            classifier.fit(X_train, t_train_bi, X_val, y_val, eta, epochs, tol, n_epochs_no_update)
        return self.classifiers

    def predict_probability(self, X: np.array) -> np.array:
        """
        A method for predicting probabilities for each class

        Parameters
        ----------
        X : array
            An input array

        Returns
        -------
        np.array
            An array of probabilities
        """
        return np.array([classifier.predict_probability(X) for classifier in self.classifiers]).T

    def predict(self, X: np.array) -> np.array:
        """
        A method for prediction

        Parameters
        ----------
        X : array
            An input array

        Returns
        -------
        np.array
            Classification of predicted values
        """
        return np.argmax(self.predict_probability(X), axis=1)
