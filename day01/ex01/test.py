import numpy as np
from vec_gradient import simple_gradient


x = np.array([12.4956442, 21.5007972, 31.5527382,
             48.9145838, 57.5088733]).reshape((-1, 1))
y = np.array([37.4013816, 36.1473236, 45.7655287,
             46.6793434, 59.5585554]).reshape((-1, 1))

# Example 0:
theta1 = np.array([2, 0.7]).reshape((-1, 1))
print(simple_gradient(x, y, theta1))
# Output: array([[-19.0342...], [-586.6687...]])

# Example 1:
theta2 = np.array([1, -0.4]).reshape((-1, 1))
print(simple_gradient(x, y, theta2))
# Output: array([[-57.8682...], [-2230.1229...]])
