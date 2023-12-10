import numpy as np
from loss import loss_

n = 7
y = (np.ones(n)).reshape(-1, 1)
y_hat = (np.zeros(n)).reshape(-1, 1)
print(loss_(y, y_hat))
# loss_ has to return 0.5

y = (np.ones(n)).reshape(-1, 1) + 4
y_hat = (np.zeros(n)).reshape(-1, 1)
print(loss_(y, y_hat))
# loss_ has to return 12.5

y = (np.ones(7)).reshape(-1, 1) + 4
y_hat = (np.arange(7)).reshape(-1, 1)
print(loss_(y, y_hat))
# loss_ has to return 4
