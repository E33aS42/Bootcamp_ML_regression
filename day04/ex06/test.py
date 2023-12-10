import numpy as np
from ridge import MyRidge as MyLR

print(" * Example 1:\n")
X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLR(np.array([[1.], [1.], [1.], [1.], [1.]]))

print(f"get_params_: \n{mylr.get_params_()}\n")
y_hat = mylr.predict_(X)
print(f"predict_: \n{y_hat}\n")
print(f"loss_elem_:\n{mylr.loss_elem_(Y, y_hat)}\n")
print(f"loss: \n{mylr.loss_(Y, y_hat)}\n")
print("fit_ with resulting thetas coefficients:")
mylr.fit_(X, Y)
print(mylr.theta)

print(" * Example 2:\n")
alpha = 1.6e-4
max_iter = 200000
print("set_param_:")
mylr.set_params_(theta=np.array([[2.], [0.], [-1.], [2.], [1.]]), alpha=alpha, max_iter=max_iter)
print(f"get_params_: \n{mylr.get_params_()}\n")
print("fit_ with resulting thetas coefficients:")
mylr.fit_(X, Y)
print(mylr.theta)

y_hat = mylr.predict_(X)
print(f"\npredict_: \n{y_hat}\n")
print(f"loss_elem_:\n{mylr.loss_elem_(Y, y_hat)}\n")
print(f"loss: \n{mylr.loss_(Y, y_hat)}\n")

