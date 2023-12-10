import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
    The matrix of polynomial features as a numpy.array, of dimension m * n,
    containing the polynomial feature values for all training examples.
    None if x is an empty numpy.array.
    None if x or power is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(x, np.ndarray) and (
            x.ndim >= 1), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
        assert isinstance(
            power, int) and power > 0, "2nd argument power must be a positive int"
        if (x.ndim == 1):
            x = x.reshape(-1, 1)
        r = x[:]
        if (power > 1):
            for i in range(2, power + 1):
                r = np.concatenate((r, x**i), axis=1)
        return r

    except Exception as e:
        print(e)
        return None
