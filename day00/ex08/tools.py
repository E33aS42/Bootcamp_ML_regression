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
        assert isinstance(x, np.ndarray) and (x.ndim == 1 or x.ndim ==
                                              2), "1st argument must be a numpy.ndarray of dimension m * n"
        assert np.any(x), "argument cannot be an empty numpy.ndarray"
        size = x.shape[0]
        ox = np.ones(size).reshape((size, 1))
        if x.ndim == 1:
            return np.concatenate((ox, x.reshape((x.shape[0], 1))), axis=1)
        else:
            return np.concatenate((ox, x), axis=1)

    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None
