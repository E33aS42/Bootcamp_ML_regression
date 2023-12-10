import numpy as np
from gradient import gradient


x = np.ones(10).reshape(-1, 1)
theta = np.array([[1], [1]])
y = np.ones(10).reshape(-1, 1)
print(gradient(x, y, theta))
# The function should returned: array([[1], [1]])

x = (np.arange(1, 25)).reshape(-1, 2)
theta = np.array([[3], [2], [1]])
y = np.arange(1, 13).reshape(-1, 1)
print(gradient(x, y, theta))
# the function should return: array([[33.5], [521.16666667], [554.66666667]])

x = (np.arange(1, 13)).reshape(-1, 3)
theta = np.array([[5], [4], [-2], [1]])
y = np.arange(9, 13).reshape(-1, 1)
print(gradient(x, y, theta))
# the function should returned: array([[ 11. ], [ 90.5], [101.5], [112.5]])
