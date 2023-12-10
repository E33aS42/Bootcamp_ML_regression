import numpy as np
from prediction import simple_predict

x = (np.arange(1, 13)).reshape(-1, 2)
theta = np.ones(3).reshape(-1, 1)
print(simple_predict(x, theta))
#  function should returned: array([[4.], [ 8.], [12.], [16.], [20.], [24.]])

x = (np.arange(1, 13)).reshape(-1, 3)
theta = np.ones(4).reshape(-1, 1)
print(simple_predict(x, theta))
# the function should returned: array([[ 7.], [16.], [25.], [34.]])

x = (np.arange(1, 13)).reshape(-1, 4)
theta = np.ones(5).reshape(-1, 1)
print(simple_predict(x, theta))
# the function should returned: array([[11.], [27.], [43.]])
