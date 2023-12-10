import numpy as np
from math import exp


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
    The sigmoid value as a numpy.ndarray of shape (m, 1).
    f(x) = 1 / (1 + exp(-(x - x0)))

    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(x, np.ndarray) and (x.ndim == 1 or x.ndim ==
                                              2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"

        if (x.ndim == 1):
            x = x.reshape(-1, 1)

        return np.array([1 / (1 + exp(-(xi))) for xi in x]).reshape(-1, 1)

    except Exception as e:
        print(e)
        return None
