import numpy as np
from polynomial_model_extended import add_polynomial_features

x = np.arange(1,11).reshape(5, 2)

# Example 1:
print(add_polynomial_features(x, 3))


# Example 2:
print(add_polynomial_features(x, 4))