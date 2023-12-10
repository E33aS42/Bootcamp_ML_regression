import numpy as np


def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(
            x, np.ndarray) and x.ndim == 1, "1st argument must be a numpy.ndarray, a vector of dimension m"
        assert isinstance(theta, np.ndarray) and theta.shape == (
            2, ), "2nd argument be a numpy.ndarray, a vector of dimension 2 * 1"
        assert np.any(x) or np.any(
            theta), "arguments cannot be empty numpy.ndarray"

    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None
    else:
        m = len(x)
        return float(theta[0]) + float(theta[1]) * x
