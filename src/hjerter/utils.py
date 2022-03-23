import numpy as np


def one_hot(a: np.ndarray, num_classes: int) -> np.ndarray:
    """One hot encoding of a 1-D array

    Args:
        a (np.ndarray): (N, ) the array with indexes
        num_classes (int): # of classes

    Returns:
        np.ndarray: (N, num_classes), the values are either 1 or 0
    """
    a = a.reshape(-1)
    a = np.eye(num_classes)[a]
    return np.squeeze(a)
