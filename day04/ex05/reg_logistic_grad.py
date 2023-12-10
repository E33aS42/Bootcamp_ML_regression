import numpy as np
from log_pred import logistic_predict_, add_intercept


def reg_logistic_grad(y, x, theta, lambda_):
	"""Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops. The three array
	Args:
	y: has to be a numpy.ndarray, a vector of shape m * 1.
	x: has to be a numpy.ndarray, a matrix of dimesion m * n.
	theta: has to be a numpy.ndarray, a vector of shape n * 1.
	lambda_: has to be a float.
	Returns:
	A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles shapes.
	Raises:
	This function should not raise any Exception.
	"""
	try:
		assert isinstance(
			x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
		assert isinstance(
			y, np.ndarray) and (y.ndim == 1 or y.ndim == 2),  "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
		assert isinstance(theta, np.ndarray) and (theta.ndim == 1 or theta.ndim == 2), "3rd argument be a numpy.ndarray, a vector of dimension 2 * 1"
		assert y.shape[0] == x.shape[0], "arrays must be the same size"
		assert isinstance(lambda_, (float, int)), "4th argument must be either a float or int"
		# assert np.any(x) or np.any(y) or np.any(theta), "arguments cannot be empty numpy.ndarray"
		if (x.ndim == 1):
			x.reshape(-1, 1)
		if (y.ndim == 1):
			y.reshape(-1, 1)
		if theta.ndim == 1:
			theta.reshape(-1, 1)

		m = y.shape[0]
		h = logistic_predict_(x, theta)

		y_hat = h.reshape(-1, 1)

		J0 = np.array([1 / m * sum([float(y_hati - yi) for y_hati, yi in zip(y_hat, y)])]).reshape(-1, 1)
		J1 = np.array(1 / m * sum([float((y_hati - yi)) * xi for y_hati, yi, xi in zip(y_hat, y, x)])).reshape(-1, 1)

		grad = np.zeros(theta.shape)
		grad[0] = J0
		grad[1:] = J1 + lambda_* theta[1:] / m
		return grad

	except Exception as e:
		print(e)
		return None

def vec_reg_logistic_grad(y, x, theta, lambda_):
	"""Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop. The three arr
	Args:
	y: has to be a numpy.ndarray, a vector of shape m * 1.
	x: has to be a numpy.ndarray, a matrix of shape m * n.
	theta: has to be a numpy.ndarray, a vector of shape n * 1.
	lambda_: has to be a float.
	Returns:
	A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles shapes.
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
		assert isinstance(theta, np.ndarray) and (theta.shape == (n + 1, 1) or theta.shape == (
				n + 1, )), "theta must be a numpy.ndarray, a vector of dimension (n + 1) * 1"
		assert y.shape[0] == x.shape[0], "arrays must be the same size"
		assert isinstance(lambda_, (float, int)), "4th argument must either be a float or int"
		if (x.ndim == 1):
			x = x.reshape(-1, 1)
		if (y.ndim == 1):
			y = y.reshape(-1, 1)
		assert y.shape[0] == x.shape[0], "arrays must be the same size"
		# assert np.any(x) or np.any(y), "arguments cannot be empty numpy.ndarray"

		h = logistic_predict_(x, theta)
		y_hat = h.reshape(-1, 1)
		sub = y_hat - y
		X = add_intercept(x)
		thet = np.copy(theta)
		thet[0] = 0
		return (np.matmul(X.T, sub) + lambda_ * thet) / m

	except Exception as e:
		print(e)
		return None
