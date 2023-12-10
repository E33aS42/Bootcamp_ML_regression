import numpy as np
import matplotlib.pyplot as plt

class MyLinearRegression():
	"""
	Description:
	My personnal linear regression class to fit like a boss.
	"""

	def __init__(self, theta, alpha=1.0e-5, max_iter=1000):
		try:
			assert isinstance(theta, np.ndarray) and theta.ndim == 2 and theta.shape[1] == 1, "1st argument theta must be a numpy.ndarray, a vector of dimension n * 1"
			assert np.any(theta), "theta cannot be an empty numpy.ndarray"
			assert isinstance(
				alpha, float), "2nd argument alpha must be a float"
			assert isinstance(
				max_iter, int) and max_iter > 0, "3rd argument max_iter must be a positive int"

			self.alpha = alpha
			self.max_iter = max_iter
			self.theta = theta

		except Exception as e:
			print(e)
			return None

	def add_intercept(self, x):
		"""Adds a column of 1 s to the non-empty numpy.array x.
		Args:
		x: has to be a numpy.array of dimension m * n.
		Returns:
		X, a numpy.array of dimension m * (n + 1).
		None if x is not a numpy.array.
		None if x is an empty numpy.array.
		Raises:
		This function should not raise any Exception.
		"""
		try:
			assert isinstance(
				x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
			# assert np.any(x), "argument cannot be an empty numpy.ndarray"
			m = x.shape[0]
			ox = np.ones(m).reshape((m, 1))
			if x.ndim == 1:
				return np.concatenate((ox, x.reshape((m, 1))), axis=1)
			else:
				return np.concatenate((ox, x), axis=1)

		except Exception as e:
			print(e)
			return None

	def predict_(self, x):
		"""Computes the prediction vector y_hat from two non-empty numpy.array.
		Args:
		x: has to be an numpy.array, a vector of dimensions m * n.
		theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
		Return:
		y_hat as a numpy.array, a vector of dimensions m * 1.
		None if x or theta are empty numpy.array.
		None if x or theta dimensions are not appropriate.
		None if x or theta is not of expected type.
		Raises:
		This function should not raise any Exception.
		"""
		try:
			assert isinstance(
				x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
			if (x.ndim == 1):
				x = x.reshape(-1, 1)
			m = x.shape[0]
			n = x.shape[1]
			assert np.any(x), "arguments cannot be empty numpy.ndarray"
			X = self.add_intercept(x)
			if X is not None:
				return np.matmul(X, self.theta)
			else:
				return None

		except Exception as e:
			print(e)
			return None

	def gradient(self, x, y):
		"""Computes a gradient vector from three non-empty numpy.array, without any for-loop.
		The three arrays must have the compatible dimensions.
		Args:
		x: has to be an numpy.array, a matrix of dimension m * n.
		y: has to be an numpy.array, a vector of dimension m * 1.
		theta: has to be an numpy.array, a vector (n +1) * 1.
		Return:
		The gradient as a numpy.array, a vector of dimensions n * 1,
		containg the result of the formula for all j.
		None if x, y, or theta are empty numpy.array.
		None if x, y and theta do not have compatible dimensions.
		None if x, y or theta is not of expected type.
		Raises:
		This function should not raise any Exception.
		"""
		try:
			assert isinstance(
				x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
			assert isinstance(
				y, np.ndarray) and (y.ndim == 1 or y.ndim == 2),  "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"

			if (x.ndim == 1):
				x = x.reshape(-1, 1)
			if (y.ndim == 1):
				y = y.reshape(-1, 1)
			assert y.shape[0] == x.shape[0], "arrays must be the same size"
			assert np.any(x) or np.any(y), "arguments cannot be empty numpy.ndarray"


			m = y.shape[0]
			h = self.predict_(x)
			y_hat = h.reshape(-1, 1)
			sub = y_hat - y
			X = self.add_intercept(x)
			return np.matmul(X.T, sub) / m

		except Exception as e:
			print(e)
			return None

	def fit_(self, x, y):
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
			assert isinstance(
				x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
			assert isinstance(
				y, np.ndarray) and (y.ndim == 1 or y.ndim == 2),  "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"

			if (x.ndim == 1):
				x = x.reshape(-1, 1)
			if (y.ndim == 1):
				y = y.reshape(-1, 1)
			assert y.shape[0] == x.shape[0], "arrays must be the same size"
			assert np.any(x) or np.any(y), "arguments cannot be empty numpy.ndarray"
			m = x.shape[0]
			n = x.shape[1]
			step = 0
			while step < self.max_iter:
				# 1. compute loss J:
				J = self.gradient(x, y)
				# 2. update theta:
				self.theta = self.theta - self.alpha * J
				step += 1

			return self.theta

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

		except Exception as e:
			print(e)
			return None

	def loss_(self, y, y_hat):
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

		except Exception as e:
			print(e)
			return None

	def mse_(self, y, y_hat):
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
			return float(sum(self.loss_elem_(y, y_hat)) / y.shape[0])

		except Exception as e:
			print(e)
			return None

	@staticmethod
	def rscore_(y, y_hat):
		"""
		Description:
		Calculate the coefficient of determination between the predicted output and the real output.
		https://openclassrooms.com/fr/courses/4297211-evaluez-les-performances-dun-modele-de-machine-learning/4308276-evaluez-un-algorithme-de-regression

		Proportion of the variation in the dependent variable that is predictable from the independent variable(s).
		It provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model.
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
			m = np.mean(y)
			rse = sum([(yi - yhi)**2 for yi, yhi in zip(y, y_hat)]) / sum([(yi - m)**2 for yi in y])
			return (1 - rse)[0]

		except Exception as e:
			print(e)
			return None

	@staticmethod
	def plot_data(x, y, Y_model, labelx=""):
		try:
			plt.scatter(x, y, marker='o', c='b', label="Sell price")
			plt.scatter(x, Y_model, marker='x', c='r',
						label="Predicted sell price")
			plt.xlabel(labelx)
			plt.ylabel("y: sell price (in keuros)")
			plt.legend()
			plt.show()

		except Exception as e:
			print(e)
			return None
