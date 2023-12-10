import matplotlib.pyplot as plt
import numpy as np
from tools import add_intercept
from prediction import predict_
from math import sqrt


def loss_elem_(y, y_hat):
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_elem: numpy.array, a vector of dimension (number of the training examples,1).
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(
            y, np.ndarray) and y.ndim == 2, "1st argument must be a numpy.ndarray, a vector of dimension m"
        assert isinstance(
            y_hat, np.ndarray) and y_hat.ndim == 2, "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert y.shape[1] == 1 and y_hat.shape[1] == 1, "arrays must be vectors of dimension m * 1"
        assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"

        return np.array([(yhi - yi)**2 for yhi, yi in zip(y_hat, y)])

    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None


def loss_2(y, y_hat):
    """
    Description:
    Calculates the value of loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_value : has to be a float.
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(
            y, np.ndarray) and y.ndim == 2, "1st argument must be a numpy.ndarray, a vector of dimension m"
        assert isinstance(
            y_hat, np.ndarray) and y_hat.ndim == 2, "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert y.shape[1] == 1 and y_hat.shape[1] == 1, "arrays must be vectors of dimension m * 1"
        assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"

        sq_dist = loss_elem_(y, y_hat)
        m = y.shape[0]
        return float(1 / (2*m) * sum(sq_dist))

    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None
