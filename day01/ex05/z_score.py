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
        assert isinstance(x, np.ndarray) and (x.ndim == 1 or x.ndim ==
                                              2), "argument must be a numpy.ndarray, a vector of dimension m"
        assert np.any(x), "argument cannot be an empty numpy.ndarray"
        size = x.shape[0]

        mu = np.mean(x)
        std = np.std(x)
        x_ = np.array([(xi - mu) / std for xi in x])
        if x.ndim == 2:
            return x_.reshape((-1,))
        return x_

    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None
