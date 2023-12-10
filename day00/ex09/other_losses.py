import numpy as np
from math import sqrt


def mse_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    try:
        assert isinstance(y, np.ndarray) and (y.ndim == 1 or y.ndim ==
                                              2), "1st argument must be a numpy.ndarray, a vector of dimension m"
        assert isinstance(y_hat, np.ndarray) and (y_hat.ndim == 1 or y_hat.ndim ==
                                                  2), "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)
        assert y.shape[1] == 1 and y_hat.shape[1] == 1, "arrays must be vectors of dimension m * 1"
        assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"
        return float(sum([(yhi - yi)**2 for yhi, yi in zip(y_hat, y)]) / y.shape[0])

    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None


def rmse_(y, y_hat):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    try:
        assert isinstance(y, np.ndarray) and (y.ndim == 1 or y.ndim ==
                                              2), "1st argument must be a numpy.ndarray, a vector of dimension m"
        assert isinstance(y_hat, np.ndarray) and (y_hat.ndim == 1 or y_hat.ndim ==
                                                  2), "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)
        assert y.shape[1] == 1 and y_hat.shape[1] == 1, "arrays must be vectors of dimension m * 1"
        assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"
        return sqrt(mse_(y, y_hat))

    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None


def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    try:
        assert isinstance(y, np.ndarray) and (y.ndim == 1 or y.ndim ==
                                              2), "1st argument must be a numpy.ndarray, a vector of dimension m"
        assert isinstance(y_hat, np.ndarray) and (y_hat.ndim == 1 or y_hat.ndim ==
                                                  2), "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)
        assert y.shape[1] == 1 and y_hat.shape[1] == 1, "arrays must be vectors of dimension m * 1"
        assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"

        return float(sum([abs(yhi - yi) for yhi, yi in zip(y_hat, y)]) / y.shape[0])

    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None


def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    try:
        assert isinstance(y, np.ndarray) and (y.ndim == 1 or y.ndim ==
                                              2), "1st argument must be a numpy.ndarray, a vector of dimension m"
        assert isinstance(y_hat, np.ndarray) and (y_hat.ndim == 1 or y_hat.ndim ==
                                                  2), "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)
        assert y.shape[1] == 1 and y_hat.shape[1] == 1, "arrays must be vectors of dimension m * 1"
        assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"

        mu = float(sum([yi for yi in y]) / y.shape[0])
        return 1 - float(sum([(yhi - yi)**2 for yhi, yi in zip(y_hat, y)])) / float(sum([(yi - mu)**2 for yi in y]))

    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None
