import numpy as np
from prediction import simple_predict

x = np.arange(1, 6)

# Example 1:
theta1 = np.array([5, 0])
print(simple_predict(x, theta1))
# Ouput: array([5., 5., 5., 5., 5.])
# Do you understand why y_hat contains only 5’s here?
# because theta1[1] = 0 multiplies every element of x by 0. Only theta1[0] = 5  is left

# Example 2:
theta2 = np.array([0, 1])
print(simple_predict(x, theta2))
# Output: array([1., 2., 3., 4., 5.])
# Do you understand why y_hat == x here?
# because theta1[1] = 1 multiplies every element of x by 1 which gives x. And theta1[0] = 0

# Example 3:
theta3 = np.array([5, 3])
print(simple_predict(x, theta3))
# Output:array([ 8., 11., 14., 17., 20.])

# Example 4:
theta4 = np.array([-3, 1])
print(simple_predict(x, theta4))
# Output: array([-2., -1., 0., 1., 2.])

# Error tests
x = [[1, 2], [3, 4]]
theta = np.array([-3, 1])
simple_predict(x, theta)

x = np.array([[1, 2], [3, 4]])
simple_predict(x, theta)

x = np.array([1, 2, 3, 4])
theta = np.array([-3, 1, 3])
simple_predict(x, theta)
