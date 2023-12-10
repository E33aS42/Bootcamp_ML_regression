import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR

# Read the dataset from the csv file
data = pd.read_csv("are_blue_pills_magic.csv")
# print(data)
Xpill = np.array(data['Micrograms']).reshape(-1, 1)
Yscore = np.array(data['Score']).reshape(-1, 1)
linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)

print(MyLR.mse_(Yscore, Y_model1))
# 57.60304285714282
print(mean_squared_error(Yscore, Y_model1))
# 57.603042857142825
print(MyLR.mse_(Yscore, Y_model2))
# 232.16344285714285
print(mean_squared_error(Yscore, Y_model2))
# 232.16344285714285

# Perform a linear regression
linear_model1.fit_(Xpill, Yscore)
linear_model2.fit_(Xpill, Yscore)
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)

# calculate the MSE
print(MyLR.mse_(Yscore, Y_model1))
print(mean_squared_error(Yscore, Y_model1))
print(MyLR.mse_(Yscore, Y_model2))
print(mean_squared_error(Yscore, Y_model2))

plt.figure(figsize=(9, 3))
#  A graph with the data and the hypothesis you get for the spacecraft piloting score versus the quantity of "blue pills"
MyLR.plot_data(Xpill, Yscore, Y_model1)

plt.figure(figsize=(9, 3))
# # Plot: The loss function J(θ) in function of the θ values
MyLR.plot_loss(Xpill, Yscore, 100)
