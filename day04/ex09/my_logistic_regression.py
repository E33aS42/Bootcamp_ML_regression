import numpy as np
import matplotlib.pyplot as plt
from math import log, exp


class MyLogisticRegression():
	"""
	Description:
	My personnal logistic regression to classify things.
	"""
	supported_penalities = ['l2'] # We consider l2 penality only. One may wants to implement other penalities

	def __init__(self, theta, alpha=0.001, max_iter=1000, penality='l2', lambda_=1.0):
		try:
			assert isinstance(theta, np.ndarray) and theta.ndim == 2 and theta.shape[1] == 1, "1st argument theta must be a numpy.ndarray, a vector of dimension n * 1"
			assert np.any(theta), "theta cannot be an empty numpy.ndarray"
			assert isinstance(
				alpha, float), "2nd argument alpha must be a float"
			assert isinstance(
				max_iter, int) and max_iter > 0, "3rd argument max_iter must be a positive int"
			assert isinstance(lambda_, (float, int)), "lambda_ must either be a float or int"

			self.alpha = alpha
			self.max_iter = max_iter
			self.theta = theta
			self.penality = penality
			self.lambda_ = lambda_ if penality in self.supported_penalities else 0

		except Exception as e:
			print(e)
			return None

	def get_params_(self):
		return self.theta, self.alpha, self.max_iter, self.penality, self.lambda_

	def set_params_(self, theta, alpha=0.001, max_iter=1000, penality='l2', lambda_=1.0):
		try:
			assert isinstance(theta, np.ndarray) and theta.ndim == 2 and theta.shape[1] == 1, "1st argument theta must be a numpy.ndarray, a vector of dimension n * 1"
			assert np.any(theta), "theta cannot be an empty numpy.ndarray"
			assert isinstance(
				alpha, float), "2nd argument alpha must be a float"
			assert isinstance(
				max_iter, int) and max_iter > 0, "3rd argument max_iter must be a positive int"
			assert isinstance(lambda_, (float, int)), "lambda_ must either be a float or int"

			self.alpha = alpha
			self.max_iter = max_iter
			self.theta = theta
			self.penality = penality
			self.lambda_ = lambda_ if penality in self.supported_penalities else 0

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
		"""Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
		Args:
		x: has to be an numpy.ndarray, a vector of dimension m * n.
		theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
		Returns:
		y_hat as a numpy.ndarray, a vector of dimension m * 1.
		None if x or theta are empty numpy.ndarray.
		None if x or theta dimensions are not appropriate.
		Raises:
		This function should not raise any Exception.
		"""
		try:
			assert isinstance(x, np.ndarray) and x.ndim >= 1, "1st argument must be a numpy.ndarray, a vector of dimension m * n"

			if (x.ndim == 1):
				x = x.reshape(-1, 1)
			assert isinstance(self.theta, np.ndarray) and self.theta.ndim == 2 and self.theta.shape[0] == x.shape[1] + 1, "2nd argument must be a numpy.ndarray, a vector of dimension n + 1"
			# assert np.any(x) and np.any(theta), "arguments cannot be empty numpy.ndarray"
			if (x.ndim == 1):
				x = x.reshape(-1, 1)
			X_ = self.add_intercept(x)
			if X_ is not None:
				prod = np.matmul(X_, self.theta)
				return np.array([round(1 / (1 + exp(-(xi)))) for xi in prod]).reshape(-1, 1)
			else:
				return None
			
		except Exception as e:
			print(e)
			return None

	def loss_elem_(self, y, y_hat, eps=1e-15):
		"""
		Computes the loss elements of the loss.
		"""
		try:
			assert isinstance(
				y, np.ndarray) and y.ndim == 2, "1st argument must be a numpy.ndarray, a vector of dimension m"
			assert isinstance(
				y_hat, np.ndarray) and y_hat.ndim == 2, "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
			assert y.shape[1] == 1 and y_hat.shape[1] == 1, "arrays must be vectors of dimension m * 1"
			assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"
			m = y.shape[0]
			return -1 / m * (np.array([y[i] * log(y_hat[i] + eps) + (1 - y[i]) * log(1 - y_hat[i] + eps) for i in range(m)]))

		except Exception as e:
			print(e)
			return None

	def loss_(self, y, y_hat, eps=1e-15):
		"""
		Computes the logistic loss value.
		Args:
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
		eps: has to be a float, epsilon (default=1e-15)
		Returns:
		The logistic loss value as a float.
		None on any error.
		Raises:
		This function should not raise any Exception.
		"""
		try:
			assert isinstance(y, np.ndarray) and (y.ndim == 1 or y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1" 
			if (y.ndim == 1):
				y = y.reshape(-1, 1)
			assert isinstance(y_hat, np.ndarray) and (y_hat.ndim == 1 or y_hat.ndim == 2), "2nd argument must be a numpy.ndarray, a vector of dimension m * 1" 
			if (y_hat.ndim == 1):
				y_hat = y_hat.reshape(-1, 1)
			assert isinstance(self.theta, np.ndarray) and (self.theta.ndim == 1 or self.theta.ndim == 2), "3rd argument must be a numpy.ndarray, a vector of dimension m * 1" 
			if (self.theta.ndim == 1):
				self.theta = self.theta.reshape(-1, 1)
			assert isinstance(self.lambda_, (int, float)), "4th argument must be a float or int"
			assert isinstance(eps, float) and eps > 0, "3rd argument must be a positive float"
			assert y.shape[1] == 1 and y_hat.shape[1] == 1, "arrays must be vectors of dimension m * 1"
			assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"
			m = y.shape[0]
			v_ones = np.ones(y.shape)
			return -1 / m * float(np.matmul(y.T, np.log(y_hat + eps * v_ones)) + np.matmul((v_ones - y).T, np.log(v_ones - y_hat + eps * v_ones)) - self.lambda_ / 2 * np.matmul(self.theta[1:].T, self.theta[1:]))

		except Exception as e:
			print(e)
			return None


	def gradient(self, x, y):
		"""Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatibl
		Args:
		x: has to be an numpy.ndarray, a matrix of shape m * n.
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
		lambda_: has to be a float.
		Returns:
		The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
		None if x, y, or theta are empty numpy.ndarray.
		None if x, y and theta do not have compatible dimensions.
		Raises:
		This function should not raise any Exception.
		"""
		try:
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
			assert isinstance(self.theta, np.ndarray) and (self.theta.shape == (n + 1, 1) or self.theta.shape == (
				n + 1, )), "theta must be a numpy.ndarray, a vector of dimension (n + 1) * 1"
			assert y.shape[0] == x.shape[0], "arrays must be the same size"
			if self.theta.shape == (n + 1, ):
				self.theta = self.theta.reshape(-1, 1)

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
