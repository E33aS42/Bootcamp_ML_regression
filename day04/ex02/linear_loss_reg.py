import numpy as np

def reg_loss_(y, y_hat, theta, lambda_):
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
		assert isinstance(y, np.ndarray) and (y.ndim == 1 or y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1" 
		if (y.ndim == 1):
			y = y.reshape(-1, 1)
		assert isinstance(y_hat, np.ndarray) and (y_hat.ndim == 1 or y_hat.ndim == 2), "2nd argument must be a numpy.ndarray, a vector of dimension m * 1" 
		if (y_hat.ndim == 1):
			y_hat = y_hat.reshape(-1, 1)
		assert isinstance(theta, np.ndarray) and (theta.ndim == 1 or theta.ndim == 2), "3rd argument must be a numpy.ndarray, a vector of dimension m * 1" 
		if (theta.ndim == 1):
			theta = theta.reshape(-1, 1)
		assert isinstance(lambda_, (int, float)), "4th argument must be a float or int"

		m = y.shape[0]
		sub = y_hat - y
		return float(1 / (2 * m) * (np.matmul(sub.T, sub) + lambda_ * np.matmul(theta[1:].T, theta[1:])))
		
	except Exception as e:
		print(e)
		return None
