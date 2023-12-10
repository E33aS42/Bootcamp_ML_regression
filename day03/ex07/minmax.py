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
        assert isinstance(x, np.ndarray) and (
            x.ndim >= 1), "argument must be a numpy.ndarray, a vector of dimension m * n"
        assert np.any(x), "argument cannot be an empty numpy.ndarray"
        m, n = x.shape
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        minx = np.min(x, axis=0)
        maxx = np.max(x, axis=0)
        x_ = (x - minx) / (maxx - minx)
        return x_, minx, maxx

    except Exception as e:
        print(e)
        return None


def de_minmax(x, minx, maxx):
    """Denormalize a previously normalized non-empty numpy.ndarray using the min-max standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x_ as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldnt raise any Exception.
    """
    try:
        assert isinstance(x, np.ndarray) and (
            x.ndim >= 1), "argument must be a numpy.ndarray, a vector of dimension m * n"
        assert np.any(x), "argument cannot be an empty numpy.ndarray"
        m, n = x.shape
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        assert len(minx) == x.shape[1] and len(
            maxx) == x.shape[1], "2nd and 3rd arguments must be numpy.ndarray of dimension n if 1st argument is of dimension m * n"
        x_ = x * (maxx - minx) + minx
        return x_

    except Exception as e:
        print(e)
        return None
