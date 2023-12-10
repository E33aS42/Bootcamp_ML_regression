import numpy as np
import matplotlib.pyplot as plt
from data_spliter import data_spliter


x = np.ones(42).reshape((-1, 1))
y = np.ones(42).reshape((-1, 1))
ret = data_spliter(x, y, 0.42)
print(list(map(np.shape, ret)))
# [(17,1), (25,1), (17,1), (25,1)]
np.random.seed(42)
tmp = np.arange(0, 110).reshape(11, 10)
x = tmp[:, :-1]
y = tmp[:, -1].reshape((-1, 1))
ret = data_spliter(x, y, 0.42)
print(ret)
