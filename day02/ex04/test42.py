import numpy as np
from fit import fit_
from prediction import predict_


x = np.arange(1, 13).reshape(-1, 3)
theta = np.array([[5], [4], [-2], [1]])
y = np.arange(9, 13).reshape(-1, 1)
alpha = 1e-2
max_iter = 10000
theta = fit_(x, y, theta, alpha, max_iter)
print(theta)
# The function should returned a result similar to:
# array([[ 7.111..],[ 1.0],[-2.888..],[ 2.222..]]) #for new_theta
# (you can check via predict_(x, new_theta) that predicted values are close to y)

x = np.arange(1, 31).reshape(-1, 6)
theta = np.array([[4], [3], [-1], [-5], [-5], [3], [-2]])
y = np.array([[128], [256], [384], [512], [640]])
alpha = 1e-4
max_iter = 42000
theta = fit_(x, y, theta, alpha, max_iter)
print(theta)
# the function should returned:
# array([[ 7.01801797], [ 0.17717732], [-0.80480472], [-1.78678675], [ 1.23123121], [12.24924918],[10.26726714]])
