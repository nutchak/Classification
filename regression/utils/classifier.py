import numpy as np


def classifier(y_predicted):
    labels = np.argmax(y_predicted, axis=1)
    return labels.reshape(-1)
