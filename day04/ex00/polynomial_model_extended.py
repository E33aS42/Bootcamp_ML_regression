import numpy as np


def add_polynomial_features(x, power):
	"""Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power give
	Args:
	x: has to be an numpy.ndarray, a matrix of shape m * n.
	power: has to be an int, the power up to which the columns of matrix x are going to be raised.
	Returns:
	The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containing the polynomial feature va
	None if x is an empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""
	try:
		assert isinstance(x, np.ndarray) and (x.ndim >= 1), "1st argument must be a numpy.ndarray, a vector of dimension m * n" 
		assert isinstance(power, int) and power > 0, "2nd argument power must be a positive int"
		if (x.ndim == 1):
			x = x.reshape(-1, 1)
		r = x[:]
		if (power > 1):
			for i in range(2, power + 1):
				r = np.concatenate((r, x**i), axis=1)
		return r
		
	except Exception as e:
		print(e)
		return None