import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR


X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
Y = np.array([[1], [0], [1]])
thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
mylr = MyLR(thetas)

print("Example 0:\n")
Y_hat = mylr.predict_(X)
print(Y_hat)
# Output: array([[0.99930437],[1.],[1.]])

print("\nExample 1:\n")
print(mylr.loss_(Y_hat, Y))
# Output: 11.513157421577004

print("\nExample 2:\n")
mylr.fit_(X, Y)
print(mylr.theta)
# Output: array([[ 2.11826435],[ 0.10154334],[ 6.43942899],[-5.10817488],[ 0.6212541 ]])

print("\nExample 3:\n")
Y_hat = mylr.predict_(X)
print(Y_hat)
print(mylr.loss_(Y_hat, Y))
# Output: array([[0.57606717],[0.68599807],[0.06562156]])

print("\nExample 4:\n")
mylr = MyLR(thetas, alpha=0.1, max_iter=10000)
mylr.fit_(X, Y)
print(mylr.theta)
Y_hat = mylr.predict_(X)
print(mylr.loss_(Y_hat, Y))
# Output: 1.4779126923052268
