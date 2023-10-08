import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from classification.utils.accuracy import accuracy


def tuning(reg, X_train: np.array, eta_start: float, eta_stop: float, eta_step:float, epoch_start: int, epoch_stop: int, epoch_step: int) -> np.array:
    """A function for plotting given range of eta and epochs

    Parameters
    ----------
    reg :
    X_train : np.array
        An input array
    eta_start : float
        Start eta value
    eta_stop : float
        Stop eta value
    eta_step : float
        Step eta value
    epoch_start : int
        Start epochs value
    epoch_stop : int
        Stop epoch value
    epoch_step : int
        Step epoch value

    Returns
    -------
    """
    def fit_eta_epoch(eta: float, epoch: int) -> float:
        """
        A method for finding accuracies for given range of eta and epochs

        Parameters
        ----------
        eta : float
            The learning rate
        epoch : int
            The number of epochs

        Returns
        -------
        float
            Accuracy for given eta and epochs
        """
        cl_tuning = reg
        cl_tuning.fit(X_train, t2_train, eta=eta, epochs=epoch)
        accuracy_eta_epoch = accuracy(cl_tuning.predict(X_val), t2_val)
        return accuracy_eta_epoch

    etas = np.arange(eta_start, eta_stop, eta_step)
    epochs = range(epoch_start, epoch_stop, epoch_step)
    # Create meshgrid (nd.array)
    eta_grid, epoch_grid = np.meshgrid(etas, epochs)

    eta_epoch_stack = np.c_[eta_grid.ravel(), epoch_grid.ravel()]
    z = np.array([fit_eta_epoch(eta, epoch.astype(int)) for eta, epoch in tqdm.tqdm(eta_epoch_stack, total=len(eta_epoch_stack))])
    # The eta and epoch that give the maximum accuracy
    eta_epoch = eta_epoch_stack[np.argmax(z)]
    z = z.reshape(eta_grid.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(eta_grid, epoch_grid, z, levels=10)
    plt.colorbar()
    plt.xlabel('eta')
    plt.ylabel('epoch')
    plt.title('Plot of meshgrid of eta and epoch')
    plt.show()
    return eta_epoch

