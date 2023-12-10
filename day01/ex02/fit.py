import numpy as np
from tools import add_intercept
from prediction import predict_
from vec_gradient import gradient


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
    new_theta: numpy.ndarray, a vector of dimension 2 * 1.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(
            x, np.ndarray) and (x.ndim == 1 or x.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m or m * 1"
        assert isinstance(
            y, np.ndarray) and (y.ndim == 1 or y.ndim == 2), "2nd argument must be a numpy.ndarray, a vector of dimension m or m * 1"
        assert isinstance(theta, np.ndarray) and (theta.shape == (2, 1) or theta.shape == (
            2, )), "3rd argument be a numpy.ndarray, a vector of dimension 2 * 1"
        assert y.shape[0] == x.shape[0], "arrays must be the same size"
        assert np.any(x) or np.any(y) or np.any(
            theta), "arguments cannot be empty numpy.ndarray"
        assert isinstance(alpha, float), "4th argument must be a float"
        assert isinstance(
            max_iter, int) and max_iter > 0, "5th argument must be a positive int"
        if (x.ndim == 1):
            x = x.reshape(-1, 1)
        if (y.ndim == 1):
            y = y.reshape(-1, 1)
        if theta.shape == (2, ):
            theta = theta.reshape(2, 1)

        m = y.shape[0]

        step = 0
        while step < max_iter:
            # 1. compute loss J:
            h = predict_(x, theta)
            y_hat = h.reshape(-1, 1)
            sub = y_hat - y
            X = add_intercept(x)
            J = np.matmul(X.T, sub) / m
            # 2. update theta:
            theta = theta - alpha * J
            step += 1

        return theta

    except AssertionError as msg:
        print(msg)
        return None
    except Exception as e:
        print(e)
        return None
