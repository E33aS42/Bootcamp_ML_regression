import numpy as np
from polynomial_model import add_polynomial_features

x = np.arange(1, 6).reshape(-1, 1)

# Example 0:
print(add_polynomial_features(x, 3))

# Example 1:
print(add_polynomial_features(x, 6))
