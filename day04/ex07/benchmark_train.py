import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from ridge import MyRidge as MyLR
from data_splitter import data_splitter
from scaler import Minmax_Scaler, Standard_Scaler
import pickle

def plot_feature(fig, x, y, Y_model, labelx=""):
	try:
		fig.scatter(x, y, marker='o', c='b', label="True price")
		fig.scatter(x, Y_model, marker='x', c='r',
					label="Prediction")
		fig.set_xlabel(labelx)
		fig.set_ylabel("Target")
		fig.legend()

	except Exception as e:
		print(e)
		return None

def plot_pred(model, x_test, y_test, y_pred, title):
	try:
		_, fig = plt.subplots(1, 3, figsize=(20, 8))
		labels = ['weight','prod_distance','time_delivery']
		for i in range(3):
			plot_feature(fig[i], x_test[:, i], y_test, y_pred, labels[i])
		plt.suptitle(title)
		plt.grid()

	except Exception as e:
		print(e)
		return None

if __name__ == "__main__":
	try:
		# 1. Load data
		path = "space_avocado.csv"
		data = pd.read_csv(path)
		x = data[['weight','prod_distance','time_delivery']].values
		y = data[['target']].values

		# 2. split data
		(x_train, x_test_, y_train, y_test_) = data_splitter(x, y, 0.6)
		(x_test, x_cv, y_test, y_cv) = data_splitter(x_test_, y_test_, 0.5)
		
		# save split data into csv files
		columnsX = ["weight", "prod_distance", "time_delivery"]
		columnsY = ["target"]
		df = pd.DataFrame(data=x_train, columns=columnsX)
		df.to_csv("x_train.csv", header=True)
		df = pd.DataFrame(data=y_train, columns=columnsY)
		df.to_csv("y_train.csv")
		df = pd.DataFrame(data=x_cv, columns=columnsX)
		df.to_csv("x_cv.csv")
		df = pd.DataFrame(data=y_cv, columns=columnsY)
		df.to_csv("y_cv.csv")
		df = pd.DataFrame(data=x_test, columns=columnsX)
		df.to_csv("x_test.csv")
		df = pd.DataFrame(data=y_test, columns=columnsY)
		df.to_csv("y_test.csv")

		# 3. Normalization
		# Minmax
		# myScalerX = Minmax_Scaler()
		# my_ScalerX.fit(x_train)
		# X_tr = my_ScalerX.transform(x_train)
		# X_cv = my_ScalerX.transform(x_cv)
		# my_ScalerY = Minmax_Scaler()
		# my_ScalerY.fit(y_train)
		# Y_tr = my_ScalerY.transform(y_train)
		# Y_cv = my_ScalerY.transform(y_cv)

		# Zscore
		my_ScalerX = Standard_Scaler()
		my_ScalerX.fit(x_train)
		X_tr = my_ScalerX.transform(x_train)
		X_cv = my_ScalerX.transform(x_cv)
		X_te = my_ScalerX.transform(x_test)
		my_ScalerY = Standard_Scaler()
		my_ScalerY.fit(y_train)
		Y_tr = my_ScalerY.transform(y_train)
		Y_cv = my_ScalerY.transform(y_cv)
		Y_te = my_ScalerY.transform(y_test)

		# 4. choose number of features
		# n_feature = 3

		# 5. add polynomial features & train models
		####### 2 features #######
		# 'weight' & 'prod_distance'
		# We choose to ignore time_delivery as it looks like it is just noise data
		n_feature = 2
		models2 = {}
		thetas = {}


#  load previously saved thetas
		with open("thetas.pickle", "rb") as f:
			thetas = pickle.load(f)

		mse2tr = [[], [], [], [], [], []]
		rscore2tr = [[], [], [], [], [], []]
		for j in range(6):
			lambd_ = 2 * j / 10
			for degree in range(4, 0, -1):
				i = degree
				print(f"\n * deg.{i} : lambda_ = {lambd_} *")
				X_tr_ = add_polynomial_features(X_tr[:, :n_feature], degree)
				# models2["deg." + str(i) + " : lambda_=" + str(lambd_)] = MyLR(theta = np.ones(1 + n_feature * degree).reshape(-1,1), alpha = 1.0e-2, max_iter = 10000, lambda_ = lambd_)
				theta2 = thetas[str(i) + "_" + str(lambd_)]
				models2["deg." + str(i) + " : lambda_=" + str(lambd_)] = MyLR(theta = theta2, alpha = 1.0e-2, max_iter = 1000, lambda_ = lambd_)
				models2["deg." + str(i) + " : lambda_=" + str(lambd_)].fit_(X_tr_, Y_tr)
				print(models2["deg." + str(i) + " : lambda_=" + str(lambd_)].theta)
				# thetas[str(i) + "_" + str(lambd_)] = models2["deg." + str(i) + " : lambda_=" + str(lambd_)].theta
				y_predtr =  my_ScalerY.de_norm(models2["deg." + str(i) + " : lambda_=" + str(lambd_)].predict_(X_tr_))
				mse2tr[j].insert(0, models2["deg." + str(i) + " : lambda_=" + str(lambd_)].mse_(y_train, y_predtr))
				rscore2tr[j].insert(0, models2["deg." + str(i) + " : lambda_=" + str(lambd_)].rscore_(y_train, y_predtr))
				print(f"R score = {rscore2tr[j][0]}")
				print(f"mse = {mse2tr[j][0]}")

# save thetas for faster computing
		# with open("thetas.pickle","wb") as f:
		# 	pickle.dump(thetas, f)

		# predict and plot data

		mse2 = [[], [], [], [], [], []]
		rscore2 = [[], [], [], [], [], []]
		for j in range(6):
			lambd_ = 2 * j / 10
			for i in range(4, 0, -1):
				degree = i
				print(f"\n * deg.{i} : lambda_ = {lambd_} *")
				X_cv_ = add_polynomial_features(X_cv[:, :n_feature], degree)
				y_pred =  my_ScalerY.de_norm(models2["deg." + str(i) + " : lambda_=" + str(lambd_)].predict_(X_cv_))
				mse2[j].insert(0, models2["deg." + str(i) + " : lambda_=" + str(lambd_)].mse_(y_cv, y_pred))
				rscore2[j].insert(0, models2["deg." + str(i) + " : lambda_=" + str(lambd_)].rscore_(y_cv, y_pred))
				# if i == 3 or i == 4:
				plot_pred(models2["deg." + str(i) + " : lambda_=" + str(lambd_)], x_cv, y_cv, y_pred, "2 features:  degree " + str(degree) + " ; lambda = " + str(lambd_))
				print(f"R score = {rscore2[j][0]}")
				print(f"mse = {mse2[j][0]}")


		# # plot mse or score of determination

		tx = range(4)
		_, fig = plt.subplots(1,6, figsize=(40,10))
		for j in range(6):
			d = 2 * j / 10
			fig[j].set_xticks(tx)
			df = pd.DataFrame({'Training mse': rscore2tr[j], 'cv mse': rscore2[j]}, index=tx)
			axes = df.plot.bar(ax=fig[j], rot=0, color={'Training mse': "orange", 'cv mse': "blue"})
			axes.legend(loc=4) 
			fig[j].set_xticklabels(["1","2","3","4"])
			fig[j].set_ylim([0, 1])
			fig[j].set_xlabel("degree")
			fig[j].set_ylabel("R score")
			fig[j].set_title(r"$\lambda}$ = " + str(d))
		title = "\n".join(r"$\lambda}$ = " + str(2 * j / 10) + " :  " + str(rscore2[j]) + "\t" + str(rscore2tr[j]) for j in range(6))
		plt.suptitle("Score of determination of the models vs. penalty term\n\n" + "cv data R score                                                                             training data R score\n" + title)
		plt.subplots_adjust(top=0.75)


		tx = range(4)
		_, fig = plt.subplots(1,6, figsize=(40, 10))
		for j in range(6):
			d = 2 * j / 10
			fig[j].set_xticks(tx)
			df = pd.DataFrame({'Training mse': mse2tr[j], 'cv mse': mse2[j]}, index=tx)
			axes = df.plot.bar(ax=fig[j], rot=0, color={'Training mse': "orange", 'cv mse': "blue"})
			axes.legend(loc=1)  
			fig[j].set_xticklabels(["1","2","3","4"])
			fig[j].set_xlabel("degree")
			fig[j].set_ylabel("mse")
			fig[j].set_title(r"$\lambda}$ = " + str(d))
		title = "\n".join(r"$\lambda}$ = " + str(2 * j / 10) + " :  " + str(mse2[j]) + "\t" + str(mse2tr[j]) for j in range(6))
		plt.suptitle("MSE of the models vs. penalty term" + "\n\ncv data MSE                                                                             training data MSE\n" + title)
		plt.subplots_adjust(top=0.75)


		# 7. Saving multiple models into pickle files 
		# https://stackoverflow.com/questions/42350635/how-can-you-pickle-multiple-objects-in-one-file

		with open("models.pickle","wb") as f:
			pickle.dump(models2, f)

		plt.show()

	except Exception as e:
		print(e)

