import numpy as np


def loss_(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Return:
    The mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    None if y or y_hat is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(
            y, np.ndarray) and y.ndim == 2, "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(
            y_hat, np.ndarray) and y_hat.ndim == 2, "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert y.shape[1] == 1 and y_hat.shape[1] == 1, "arrays must be vectors of dimension m * 1"
        assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"
        m = y.shape[0]
        sub = y_hat - y
        return float(np.matmul(sub.T, sub) / (2 * m))
        # return float(1 / (2*m) * sum([(yi - yhi)**2 for yi, yhi in zip(y, y_hat)]))

    except Exception as e:
        print(e)
        return None
