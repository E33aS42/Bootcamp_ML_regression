import matplotlib.pyplot as plt
import numpy as np


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=150000):
        try:
            assert isinstance(thetas, np.ndarray) and thetas.shape == (
                2, 1), "1st argument thetas must be a numpy.ndarray, a vector of dimension 2 * 1"
            assert isinstance(
                alpha, float), "2nd argument alpha must be a float"
            assert isinstance(
                max_iter, int) and max_iter > 0, "3rd argument max_iter must be a positive int"
            if not np.any(thetas):
                thetas = np.array([[0.], [0.]])
            self.alpha = alpha
            self.max_iter = max_iter
            self.thetas = thetas.astype(float)

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
            return float(np.matmul(sub.T, sub) / (2 * m))

        except AssertionError as msg:
            print(msg)
            return None
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def plot_data(x, y, Y_model):
        plt.scatter(x, y, marker='o', c='b', label=r"$S_{true} (pills)$")
        plt.scatter(x, Y_model, marker='x', c='r',
                    label=r"$S_{predict} (pills)$")
        plt.plot(x, Y_model, 'r--')
        plt.xlabel('Quantity of blue pills (in micrograms)')
        plt.ylabel('Space driving score')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_loss(x, y, n):
        linear_model = MyLinearRegression(np.array([[0.], [0.]]))
        for t0 in np.linspace(80, 100, num=6):
            J = []
            linear_model.thetas[0][0] = np.float64(t0)
            for t1 in np.linspace(-15, -4, n):
                linear_model.thetas[1][0] = np.float64(t1)
                Y_model = linear_model.predict_(x)
                J.append(linear_model.loss_(y, Y_model))
            plt.plot(np.linspace(-14, -4, n), J, label=r"J($\theta_0$ = " +
                     f"{int(t0)}, " + r"$\theta_1$)", linewidth=2)
        plt.xlim((-14, -3))
        plt.ylim((0, 140))
        plt.legend(loc=4)
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(f"Cost function " + r"J($\theta_0$, $\theta_1$)")

        plt.grid()
        plt.show()

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
