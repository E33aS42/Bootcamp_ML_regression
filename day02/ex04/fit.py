import numpy as np
from prediction import predict_, add_intercept
from gradient import gradient


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.array, a matrix of dimension m * n:
    (number of training examples, number of features).
    y: has to be a numpy.array, a vector of dimension m * 1:
    (number of training examples, 1).
    theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
    (number of features + 1, 1).
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
    None if there is a matching dimension problem.
    None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(alpha, float), "4th argument must be a float"
        assert isinstance(
            max_iter, int) and max_iter > 0, "5th argument must be a positive int"
        assert isinstance(
            x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
        assert isinstance(
            y, np.ndarray) and (y.ndim == 1 or y.ndim == 2),  "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        m = x.shape[0]
        n = x.shape[1]
        if (x.ndim == 1):
            x = x.reshape(-1, 1)
        if (y.ndim == 1):
            y = y.reshape(-1, 1)
        assert isinstance(theta, np.ndarray) and (theta.shape == (n + 1, 1) or theta.shape == (
            n + 1, )), "3rd argument be a numpy.ndarray, a vector of dimension (n + 1) * 1"
        assert y.shape[0] == x.shape[0], "arrays must be the same size"
        assert np.any(x) or np.any(y) or np.any(
            theta), "arguments cannot be empty numpy.ndarray"
        if theta.shape == (n + 1, ):
            theta = theta.reshape(-1, 1)

        step = 0
        while step < max_iter:
            # 1. compute loss J:
            J = gradient(x, y, theta)
            # 2. update theta:
            theta = theta - alpha * J
            step += 1

        return theta

    except Exception as e:
        print(e)
        return None
