import numpy as np


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        try:
            assert isinstance(thetas, np.ndarray) and thetas.shape == (
                2, 1), "1st argument thetas must be a numpy.ndarray, a vector of dimension 2 * 1"
            assert np.any(thetas), "thetas cannot be an empty numpy.ndarray"
            assert isinstance(
                alpha, float), "2nd argument alpha must be a float"
            assert isinstance(
                max_iter, int) and max_iter > 0, "3rd argument max_iter must be a positive int"

            self.alpha = alpha
            self.max_iter = max_iter
            self.thetas = thetas

        except AssertionError as msg:
            print(msg)
            return None
        except Exception as e:
            print(e)
            return None

    def add_intercept(self, x):
        try:
            assert isinstance(x, np.ndarray) and (x.ndim == 1 or x.ndim ==
                                                  2), "1st argument must be a numpy.ndarray of dimension m * 1"
            assert np.any(x), "argument cannot be an empty numpy.ndarray"
            size = x.shape[0]
            ox = np.ones(size).reshape((size, 1))
            if x.ndim == 1:
                return np.concatenate((ox, x.reshape((x.shape[0], 1))), axis=1)
            else:
                return np.concatenate((ox, x), axis=1)

        except AssertionError as msg:
            print(msg)
            return None
        except Exception as e:
            print(e)
            return None

    def predict_(self, x):
        try:
            assert isinstance(
                x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m"
            if x.ndim == 2:
                assert x.shape[1] == 1, "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
            elif (x.ndim != 1 and x.ndim != 2):
                raise Exception(
                    "1st argument must be a numpy.ndarray, a vector of dimension m")
            assert isinstance(self.thetas, np.ndarray) and (self.thetas.shape == (2, 1) or self.thetas.shape == (
                2, )), "2nd argument be a numpy.ndarray, a vector of dimension 2 * 1"
            assert np.any(x) or np.any(
                self.thetas), "arguments cannot be empty numpy.ndarray"

            if self.thetas.shape == (2, ):
                self.thetas.reshape(2, 1)
            X = self.add_intercept(x)
            return np.matmul(X, self.thetas)

        except AssertionError as msg:
            print(msg)
            return None
        except Exception as e:
            print(e)
            return None

    def fit_(self, x, y):
        try:
            assert isinstance(
                x, np.ndarray) and (x.ndim == 1 or x.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m or m * 1"
            assert isinstance(
                y, np.ndarray) and (y.ndim == 1 or y.ndim == 2), "2nd argument must be a numpy.ndarray, a vector of dimension m or m * 1"
            assert y.shape[0] == x.shape[0], "arrays must be the same size"
            assert np.any(x) or np.any(
                y), "arguments cannot be empty numpy.ndarray"
            if (x.ndim == 1):
                x.reshape(-1, 1)
            if (y.ndim == 1):
                y.reshape(-1, 1)

            m = y.shape[0]

            step = 0
            while step < self.max_iter:
                # 1. compute loss J:
                h = self.predict_(x)
                y_hat = h.reshape(-1, 1)
                sub = y_hat - y
                X = self.add_intercept(x)
                J = np.matmul(X.T, sub) / m
                # 2. update theta:
                self.thetas = self.thetas - self.alpha * J
                step += 1

            return self.thetas

        except AssertionError as msg:
            print(msg)
            return None
        except Exception as e:
            print(e)
            return None

    def loss_elem_(self, y, y_hat):
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

    def loss_(self, y, y_hat):
        try:
            assert isinstance(
                y, np.ndarray) and y.ndim == 2, "1st argument must be a numpy.ndarray, a vector of dimension m"
            assert isinstance(
                y_hat, np.ndarray) and y_hat.ndim == 2, "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
            assert y.shape[1] == 1 and y_hat.shape[1] == 1, "arrays must be vectors of dimension m * 1"
            assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"
            sub = y_hat - y
            m = y.shape[0]
            # return float(sum(sub * sub) / (2 * m))
            return float(np.matmul(sub.T, sub) / (2 * m))

        except AssertionError as msg:
            print(msg)
            return None
        except Exception as e:
            print(e)
            return None
