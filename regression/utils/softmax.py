import numpy as np


def softmax(logits):
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=1).reshape(-1, 1)
