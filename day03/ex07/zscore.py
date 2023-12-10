import numpy as np

# A standard score is the number of standard deviations by which the value of a raw score (i.e., an observed value or data point) is above or below the mean value of what is being observed or measured.
# In practice, it helps determine the significance of a set of data.


def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
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
        size = x.shape[0]
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        mu = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x_ = x - mu
        x_ /= std
        return x_, std, mu

    except Exception as e:
        print(e)
        return None


def de_zscore(x, std, mu):
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
        assert len(std) == x.shape[1] and len(
            mu) == x.shape[1], "2nd and 3rd arguments must be numpy.ndarray of dimension n if 1st argument is of dimension m * n"
        x_ = x * std + mu
        return x_

    except Exception as e:
        print(e)
        return None
