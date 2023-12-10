import matplotlib.pyplot as plt
import numpy as np
from tools import add_intercept
from prediction import predict_


def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
    """
    try:
        assert isinstance(
            x, np.ndarray) and x.ndim == 1, "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(
            y, np.ndarray) and y.ndim == 1, "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(theta, np.ndarray) and theta.shape == (
            2, 1), "3rd argument be a numpy.ndarray, a vector of dimension 2 * 1"
        assert np.any(x) or np.any(y) or np.any(
            theta), "arguments cannot be empty numpy.ndarray"

        h = predict_(x, theta)

        fig = plt.figure()
        p1 = plt.plot(x, y, 'o', label='data')
        p2 = plt.plot(x, h, label='prediction')
        plt.title('Plot')
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
