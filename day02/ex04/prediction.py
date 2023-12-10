import numpy as np


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
        if (x.ndim < 2):
            raise Exception(
                "1st argument must be a numpy.ndarray, a vector of dimension m * n")
        assert np.any(x), "argument cannot be an empty numpy.ndarray"
        m = x.shape[0]
        ox = np.ones(m).reshape((m, 1))
        if x.ndim == 1:
            return np.concatenate((ox, x.reshape((m, 1))), axis=1)
        else:
            return np.concatenate((ox, x), axis=1)

    except Exception as e:
        print(e)
        return None


def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimensions m * n.
    theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
    Return:
    y_hat as a numpy.array, a vector of dimensions m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    None if x or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(
            x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
        if (x.ndim < 2):
            raise Exception(
                "1st argument must be a numpy.ndarray, a vector of dimension m * n")
        m = x.shape[0]
        n = x.shape[1]
        assert isinstance(theta, np.ndarray) and (theta.shape == (n + 1, 1) or theta.shape == (
            n + 1, )), "2nd argument be a numpy.ndarray, a vector of dimension (n + 1) * 1"
        assert np.any(x) or np.any(
            theta), "arguments cannot be empty numpy.ndarray"

        if theta.shape == (n + 1, ):
            theta = theta.reshape(-1, 1)

        X = add_intercept(x)
        return np.matmul(X, theta)
        # return float(theta[0]) + sum([float(theta[j + 1]) * x[:, j] for j in range(n)]).reshape(-1, 1)

    except Exception as e:
        print(e)
        return None
