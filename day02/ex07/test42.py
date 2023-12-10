import numpy as np
from polynomial_model import add_polynomial_features

x1 = np.arange(1, 6).reshape(-1, 1)
x1_poly = add_polynomial_features(x1, 5)
print(x1_poly)
# array([[ 1, 1, 1, 1, 1], [ 2, 4, 8, 16, 32], [ 3, 9, 27, 81, 243], [ 4, 16, 64, 256, 1024], [ 5, 25, 125, 625, 3125]])

x2 = np.arange(10, 40, 10).reshape(-1, 1)
x2_poly = add_polynomial_features(x2, 5)
print(x2_poly)
# array([[10, 100, 1000, 10000, 100000] [20, 400, 8000, 160000, 3200000] [30, 900, 27000, 810000, 24300000]]])

x3 = np.arange(10, 40, 10).reshape(-1, 1)/10
x3_poly = add_polynomial_features(x3, 3)
print(x3_poly)
# array([[1. , 1. , 1. ], [ 2. , 4. , 8. ], [ 3. , 9. , 27. ]])
