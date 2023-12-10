import numpy as np
from mylinearregression import MyLinearRegression as MyLR


class MyRidge(MyLR):
    """
    The main idea behind Ridge Regression is to find a new line that doesnt fit the training data well. We introduce a small amount of bias into how the new line is fit to the data.
    In return of that small of bias, we get a significant drop in variance.
    In other words, by starting with a slightly worse fit, RR can provide better long term predictions.
    And so we add a penalty (lambda * slope**2) to the traditional Least Square method. LAmbda determines how severe that penalty is.

    Note: as we increase lambda, the slope is getting assymptotically close to 0. As lmabda increases, predictions get less and less sensitive to features scale.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000, lambda_=0.5):
        try:
            assert isinstance(theta, np.ndarray) and theta.ndim == 2 and theta.shape[
                1] == 1, "1st argument theta must be a numpy.ndarray, a vector of dimension n * 1"
            assert np.any(theta), "theta cannot be an empty numpy.ndarray"
            assert isinstance(
                alpha, float), "2nd argument alpha must be a float"
            assert isinstance(
                max_iter, int) and max_iter > 0, "3rd argument max_iter must be a positive int"
            assert isinstance(lambda_, (float, int)
                              ), "lambda_ must either be a float or int"

            self.alpha = alpha
            self.max_iter = max_iter
            self.theta = theta
            self.lambda_ = lambda_

        except Exception as e:
            print(e)
            return None

    def get_params_(self):
        return self.theta, self.alpha, self.max_iter, self.lambda_

    def set_params_(self, theta, alpha=0.001, max_iter=1000, lambda_=1.0):
        try:
            assert isinstance(theta, np.ndarray) and theta.ndim == 2 and theta.shape[
                1] == 1, "1st argument theta must be a numpy.ndarray, a vector of dimension n * 1"
            assert np.any(theta), "theta cannot be an empty numpy.ndarray"
            assert isinstance(
                alpha, float), "2nd argument alpha must be a float"
            assert isinstance(
                max_iter, int) and max_iter > 0, "3rd argument max_iter must be a positive int"
            assert isinstance(lambda_, (float, int)
                              ), "lambda_ must either be a float or int"

            self.alpha = alpha
            self.max_iter = max_iter
            self.theta = theta
            self.lambda_ = lambda_

        except Exception as e:
            print(e)
            return None

    def loss_(self, y, y_hat):
        """Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.
        Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
        Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta are empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
        Raises:
        This function should not raise any Exception.
        """
        try:
            assert isinstance(y, np.ndarray) and (
                y.ndim == 1 or y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
            if (y.ndim == 1):
                y = y.reshape(-1, 1)
            assert isinstance(y_hat, np.ndarray) and (y_hat.ndim == 1 or y_hat.ndim ==
                                                      2), "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
            if (y_hat.ndim == 1):
                y_hat = y_hat.reshape(-1, 1)
            assert isinstance(self.theta, np.ndarray) and (
                self.theta.ndim == 1 or self.theta.ndim == 2), "theta must be a numpy.ndarray, a vector of dimension m * 1"
            if (self.theta.ndim == 1):
                self.theta = self.theta.reshape(-1, 1)
            assert isinstance(self.lambda_, (int, float)
                              ), "lambda must be a float or int"

            m = y.shape[0]
            sub = y_hat - y
            return float(1 / (2 * m) * (np.matmul(sub.T, sub) + self.lambda_ * np.matmul(self.theta[1:].T, self.theta[1:])))

        except Exception as e:
            print(e)
            return None

    def gradient(self, x, y):
        """Computes the regularized linear gradient of three non-empty numpy.ndarray,
        without any for-loop. The three arrays must have compatible shapes.
        Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
        Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        try:
            assert isinstance(
                x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
            assert isinstance(
                y, np.ndarray) and (y.ndim == 1 or y.ndim == 2),  "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
            assert isinstance(self.theta, np.ndarray) and (
                self.theta.ndim == 1 or self.theta.ndim == 2), "theta be a numpy.ndarray, a vector of dimension 2 * 1"
            assert y.shape[0] == x.shape[0], "arrays must be the same size"
            if (x.ndim == 1):
                x = x.reshape(-1, 1)
            if (y.ndim == 1):
                y = y.reshape(-1, 1)
            assert y.shape[0] == x.shape[0], "arrays must be the same size"
            # assert np.any(x) or np.any(y), "arguments cannot be empty numpy.ndarray"

            m = y.shape[0]
            h = self.predict_(x)
            y_hat = h.reshape(-1, 1)
            sub = y_hat - y
            X = self.add_intercept(x)
            thet = np.copy(self.theta)
            thet[0] = 0
            return (np.matmul(X.T, sub) + self.lambda_ * thet) / m

        except Exception as e:
            print(e)
            return None
