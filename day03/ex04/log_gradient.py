import numpy as np
from log_loss import log_loss_
from log_pred import logistic_predict_, add_intercept


def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatibl
    Args:
    x: has to be an numpy.ndarray, a matrix of shape m * n.
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
    The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(
            x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
        assert isinstance(
            y, np.ndarray) and (y.ndim == 1 or y.ndim == 2),  "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        m = x.shape[0]
        n = x.shape[1]
        if (x.ndim == 1):
            x = x.reshape(-1, 1)
        if (y.ndim == 1):
            y = y.reshape(-1, 1)
        assert isinstance(theta, np.ndarray) and (theta.shape == (n + 1, 1) or theta.shape == (
            n + 1, )), "3rd argument be a numpy.ndarray, a vector of dimension (n + 1) * 1"
        assert y.shape[0] == x.shape[0], "arrays must be the same size"
        if theta.shape == (n + 1, ):
            theta = theta.reshape(-1, 1)

        m = y.shape[0]
        h = logistic_predict_(x, theta)
        y_hat = h.reshape(-1, 1)
        J0 = list([1 / m * sum([float(y_hati - yi)
                  for y_hati, yi in zip(y_hat, y)])])
        J1 = list(1 / m * sum([float((y_hati - yi)) *
                  xi for y_hati, yi, xi in zip(y_hat, y, x)]))
        return np.array(J0 + J1).reshape(-1, 1)

    except Exception as e:
        print(e)
        return None
