import numpy as np
from tools import add_intercept


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """
    try:
        assert isinstance(
            x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m"
        if x.ndim == 2:
            assert x.shape[1] == 1, "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        elif (x.ndim != 1 and x.ndim != 2):
            raise Exception(
                "1st argument must be a numpy.ndarray, a vector of dimension m")
        assert isinstance(theta, np.ndarray) and (theta.shape == (2, 1) or theta.shape == (
            2, )), "2nd argument be a numpy.ndarray, a vector of dimension 2 * 1"
        assert np.any(x) or np.any(
            theta), "arguments cannot be empty numpy.ndarray"

        if theta.shape == (2, ):
            theta = theta.reshape(2, 1)
        X = add_intercept(x)
        return np.matmul(X, theta)

    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None
