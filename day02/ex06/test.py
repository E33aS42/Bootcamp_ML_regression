import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt

data = pd.read_csv("spacecraft_data.csv")

# Part 1
print("*** Part One: Univariate Linear Regression ***\n")

# Age
print("* Age feature *\n")
X = np.array(data[['Age']])
Y = np.array(data[['Sell_price']])

myLR_age = MyLR(theta=np.array([[1000.0], [-1.0]]),
                alpha=6.0e-4, max_iter=100000)
myLR_age.fit_(X.reshape(-1, 1), Y)
print("thetas: \n", myLR_age.theta)
y_pred = myLR_age.predict_(X[:, 0].reshape(-1, 1))

print("mse: ", myLR_age.mse_(y_pred, Y))
# Output: 55736.86719 for alpha = 2.5e-5

myLR_age.plot_data(X, Y, y_pred, "x1: age (in years")

# # Thrust
print("\n* Thrust feature *\n")
X = np.array(data[['Thrust_power']])
myLR_thrust = MyLR(theta=np.array(
    [[1.0], [1000.0]]), alpha=1.0e-4, max_iter=200000)
myLR_thrust.fit_(X.reshape(-1, 1), Y)
print("thetas: \n", myLR_thrust.theta)
y_pred = myLR_thrust.predict_(X[:, 0].reshape(-1, 1))
print("mse: ", myLR_thrust.mse_(y_pred, Y))
myLR_age.plot_data(X, Y, y_pred, "x2: thrust power (in 10Km/s")

# # Total distance
print("\n* Total distance feature *\n")
X = np.array(data[['Terameters']])
myLR_dist = MyLR(theta=np.array(
    [[1000.0], [-1.0]]), alpha=1.0e-4, max_iter=200000)
myLR_dist.fit_(X.reshape(-1, 1), Y)
print("thetas: \n", myLR_dist.theta)
y_pred = myLR_dist.predict_(X.reshape(-1, 1))
print("mse: ", myLR_dist.mse_(y_pred, Y))
myLR_age.plot_data(
    X, Y, y_pred, "x3: distance totalizer value of spacecraft (in Tmeters")
plt.show()

# Part 2
print("\n*** Part Two: Multivariate Linear Regression ***\n")

X = np.array(data[['Age', 'Thrust_power', 'Terameters']])
Y = np.array(data[['Sell_price']])
my_lreg = MyLR(theta=np.array(
    [[100.0], [1.0], [1.0], [1.0]]), alpha=8e-5, max_iter=600000)

my_lreg.fit_(X, Y)
print("thetas: \n", my_lreg.theta)

y_pred = my_lreg.predict_(X)

print("mse: ", my_lreg.mse_(y_pred, Y))

my_lreg.plot_data(X[:, 0], Y, y_pred, "x1: age (in years")
my_lreg.plot_data(X[:, 1], Y, y_pred, "x2: thrust power (in 10Km/s")
my_lreg.plot_data(X[:, 2], Y, y_pred,
                  "x3: distance totalizer value of spacecraft (in Tmeters")
plt.show()
