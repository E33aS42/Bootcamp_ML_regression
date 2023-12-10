import numpy as np
from tools import add_intercept
from prediction import predict_


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, with a for-loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(
            x, np.ndarray) and (x.ndim == 1 or x.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m or m * 1"
        assert isinstance(
            y, np.ndarray) and (y.ndim == 1 or y.ndim == 2), "2nd argument must be a numpy.ndarray, a vector of dimension m or m * 1"
        assert isinstance(theta, np.ndarray) and (theta.shape == (2, 1) or theta.shape == (
            2, )), "3rd argument be a numpy.ndarray, a vector of dimension 2 * 1"
        assert y.shape[0] == x.shape[0], "arrays must be the same size"
        assert np.any(x) or np.any(y) or np.any(
            theta), "arguments cannot be empty numpy.ndarray"
        if (x.ndim == 1):
            x = x.reshape(-1, 1)
        if (y.ndim == 1):
            y = y.reshape(-1, 1)
        if theta.shape == (2, ):
            theta = theta.reshape(2, 1)

        m = y.shape[0]
        h = predict_(x, theta)
        y_hat = h.reshape(-1, 1)

        J0 = 0
        J1 = 0
        for y_hati, yi, xi in zip(y_hat, y, x):
            J0 += y_hati - yi
            J1 += (y_hati - yi) * xi
        J0 = [float(J0 / m)]
        J1 = [float(J1 / m)]
        return np.array([J0, J1])


    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None
