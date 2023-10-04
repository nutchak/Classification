import numpy as np


def one_hot_encoding(t, k):
    n_labels = t.shape[0]
    t_ova = np.zeros((n_labels, k))
    for i in range(n_labels):
        label = t[i]
        t_ova[i][label] = 1
    return t_ova