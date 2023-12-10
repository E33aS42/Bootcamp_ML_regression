import numpy as np


def iterative_l2(theta):
	"""Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
	Args:
	theta: has to be a numpy.ndarray, a vector of shape n * 1.
	Returns:
	The L2 regularization as a float.
	None if theta in an empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""
	try:
		assert isinstance(theta, np.ndarray) and (theta.ndim == 1 or theta.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1" 

		if (theta.ndim == 1):
			theta = theta.reshape(-1, 1)
		return float(sum([ti**2 for ti in theta[1:]]))
		
	except Exception as e:
		print(e)
		return None

def l2(theta):
	"""Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
	Args:
	theta: has to be a numpy.ndarray, a vector of shape n * 1.
	Returns:
	The L2 regularization as a float.
	None if theta in an empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""
	try:
		assert isinstance(theta, np.ndarray) and (theta.ndim == 1 or theta.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1" 

		if (theta.ndim == 1):
			theta = theta.reshape(-1, 1)
		return float(np.matmul(theta[1:].T, theta[1:]))
		
	except Exception as e:
		print(e)
		return None