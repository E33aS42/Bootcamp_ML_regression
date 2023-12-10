import numpy as np
from prediction import predict_, add_intercept


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible dimensions.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
    The gradient as a numpy.array, a vector of dimensions n * 1,
    containg the result of the formula for all j.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible dimensions.
    None if x, y or theta is not of expected type.
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
        assert np.any(x) or np.any(y) or np.any(
            theta), "arguments cannot be empty numpy.ndarray"
        if theta.shape == (n + 1, ):
            theta = theta.reshape(-1, 1)

        m = y.shape[0]
        h = predict_(x, theta)
        y_hat = h.reshape(-1, 1)
        sub = y_hat - y
        X = add_intercept(x)
        return np.matmul(X.T, sub) / m

    except Exception as e:
        print(e)
        return None
