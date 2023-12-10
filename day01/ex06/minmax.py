import numpy as np


def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x_ as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldnt raise any Exception.
    """
    try:
        assert isinstance(x, np.ndarray) and (x.ndim == 1 or x.ndim ==
                                              2), "argument must be a numpy.ndarray, a vector of dimension m"
        assert np.any(x), "argument cannot be an empty numpy.ndarray"
        size = x.shape[0]
        mu = np.mean(x)
        std = np.std(x)
        x_ = np.array([(xi - min(x)) / (max(x) - min(x)) for xi in x])
        if x_.shape == (size, 1):
            return x_.reshape((-1,))
        return x_

    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None
