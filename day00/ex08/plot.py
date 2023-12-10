import matplotlib.pyplot as plt
import numpy as np
import warnings
from functools import wraps
from tools import add_intercept
from prediction import predict_
from vec_loss import loss_
from loss import loss_elem_, loss_2


def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner


@ignore_warnings
def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """

    try:
        assert isinstance(
            x, np.ndarray) and x.ndim == 1, "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(
            y, np.ndarray) and y.ndim == 1, "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(theta, np.ndarray) and (theta.shape == (2, 1) or theta.shape == (
            2, )), "3rd argument be a numpy.ndarray, a vector of dimension 2 * 1"
        assert np.any(x) or np.any(y) or np.any(
            theta), "arguments cannot be empty numpy.ndarray"

        if theta.shape == (2, ):
            theta.reshape(2, 1)
        h = predict_(x, theta)

        y_hat = h.reshape(-1, 1)
        Y = y.reshape(-1, 1)
        J = loss_(y_hat, Y)
        J2 = loss_2(y_hat, Y)
        # dif = np.array([yi - yhi for yhi, yi in zip(y_hat, y)])

        fig = plt.figure()

        for xi, yhi, yi in zip(x, y_hat, y):
            plt.plot([xi, xi], [yhi, yi], 'r', linestyle="--")
        p1 = plt.plot(x, y, 'o', label='data')
        p2 = plt.plot(x, h, label='prediction')
        # Why is there a factor 2 on the graphs title ? Did someone forget to divide by 2 in the loss calculation?
        plt.title(f"Cost : {2 * J2: 0.6f}")
        plt.xlabel('')
        plt.ylabel('')
        plt.legend()
        plt.grid(True)
        plt.show()

    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None
