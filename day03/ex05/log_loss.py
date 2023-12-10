import numpy as np
from math import log


def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    eps: has to be a float, epsilon (default=1e-15)
    Returns:
    The logistic loss value as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(
            y, np.ndarray) and y.ndim == 2, "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(
            y_hat, np.ndarray) and y_hat.ndim == 2, "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(
            eps, float) and eps > 0, "3rd argument must be a positive float"
        assert y.shape[1] == 1 and y_hat.shape[1] == 1, "arrays must be vectors of dimension m * 1"
        assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"
        m = y.shape[0]
        v_ones = np.ones(y.shape)
        return -1 / m * float(np.matmul(y.T, np.log(y_hat + eps * v_ones)) + np.matmul((v_ones - y).T, np.log(v_ones - y_hat + eps * v_ones)))

    except Exception as e:
        print(e)
        return None
