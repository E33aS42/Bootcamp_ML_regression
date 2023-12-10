import numpy as np


def simple_predict(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
    Return:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not matching.
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

        return float(theta[0]) + sum([float(theta[j + 1]) * x[:, j] for j in range(n)]).reshape(-1, 1)

    except Exception as e:
        print(e)
        return None
