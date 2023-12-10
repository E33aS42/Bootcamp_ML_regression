import numpy as np
from math import exp


def add_intercept(x):
    """Adds a column of 1 s to the non-empty numpy.array x.
    Args:
    x: has to be a numpy.array of dimension m * n.
    Returns:
    X, a numpy.array of dimension m * (n + 1).
    None if x is not a numpy.array.
    None if x is an empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(
            x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
        # assert np.any(x), "argument cannot be an empty numpy.ndarray"
        m = x.shape[0]
        ox = np.ones(m).reshape((m, 1))
        if x.ndim == 1:
            return np.concatenate((ox, x.reshape((m, 1))), axis=1)
        else:
            return np.concatenate((ox, x), axis=1)

    except Exception as e:
        print(e)
        return None


def logistic_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * n.
    theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(
            x, np.ndarray) and x.ndim >= 1, "1st argument must be a numpy.ndarray, a vector of dimension m * n"

        if (x.ndim == 1):
            x = x.reshape(-1, 1)
        assert isinstance(theta, np.ndarray) and theta.ndim == 2 and theta.shape[0] == x.shape[
            1] + 1, "2nd argument must be a numpy.ndarray, a vector of dimension n + 1"
        # assert np.any(x) and np.any(theta), "arguments cannot be empty numpy.ndarray"
        if (x.ndim == 1):
            x = x.reshape(-1, 1)
        X_ = add_intercept(x)
        if X_ is not None:
            prod = np.matmul(X_, theta)
            return np.array([1 / (1 + exp(-(xi))) for xi in prod]).reshape(-1, 1)
        else:
            return None

    except Exception as e:
        print(e)
        return None
