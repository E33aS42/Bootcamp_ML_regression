import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from ridge import MyRidge as MyLR
from scaler import Minmax_Scaler, Standard_Scaler
import pickle
from benchmark_train import plot_pred

def load_data(path):
	try:
		data = pd.read_csv(path)
		return data.values[:, 1:]

	except exception as e:
		print(e)
		return None

def plot_best(fig, model, x_test, y_test, y_pred):
	try:

		for i in range(3):

			fig[i].scatter(x_test[:, i], y_test, marker='o', c='b', label="True price")
			fig[i].scatter(x_test[:, i], y_pred, marker='x', c='r',
					label="Prediction")

	except Exception as e:
		print(e)
		return None

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
		# 1. Load split data
		x_features = ['weight','prod_distance','time_delivery']
		y_feature = ['target']
		x_train = load_data("x_train.csv")
		x_test = load_data("x_test.csv")
		y_train = load_data("y_train.csv")
		y_test = load_data("y_test.csv")

		# 3. Normalization
		# Minmax
		# myScaler = Minmax_Scaler()
		# myScaler.fit(x_train)
		# X_tr = myScaler.transform(x_train)
		# X_te = myScaler.transform(x_test)

		# Zscore
		my_ScalerX = Standard_Scaler()
		my_ScalerX.fit(x_train)
		X_tr = my_ScalerX.transform(x_train)
		X_te = my_ScalerX.transform(x_test)
		my_ScalerY = Standard_Scaler()
		my_ScalerY.fit(y_train)
		Y_tr = my_ScalerY.transform(y_train)

		# 4. choose number of features
		####### 2 features #######
		# 'weight' & 'prod_distance'
		# We choose to ignore time_delivery as it looks like it is just noise data
		n_feature = 2
		models_best = {}
		# best model appears to be degree 3 or 4 with lambda = 0.0
		degree = 3

		# with open("thetas_best.pickle", "rb") as f:
		# 	thetas = pickle.load(f)
		models2_4 = {}
		thetas = {}
		mse2tr = [[], [], [], [], [], []]
		rscore2tr = [[], [], [], [], [], []]
		for j in range(6):
			lambd_ = 2 * j / 10
			i = degree
			print(f"\n * deg.{i} : lambda_ = {lambd_} *")
			X_tr_ = add_polynomial_features(X_tr[:, :n_feature], degree)
			models2_4["deg." + str(i) + " : lambda_=" + str(lambd_)] = MyLR(theta = np.ones(1 + n_feature * degree).reshape(-1,1), alpha = 1.0e-2, max_iter = 10000, lambda_ = lambd_)
			# theta2 = thetas[str(i) + "_" + str(lambd_)]
			# models2_4["deg." + str(i) + " : lambda_=" + str(lambd_)] = MyLR(theta = theta2, alpha = 1.0e-2, max_iter = 1000, lambda_ = lambd_)
			models2_4["deg." + str(i) + " : lambda_=" + str(lambd_)].fit_(X_tr_, Y_tr)
			# print(models2_4["deg." + str(i) + " : lambda_=" + str(lambd_)].theta)
			thetas[str(i) + "_" + str(lambd_)] = models2_4["deg." + str(i) + " : lambda_=" + str(lambd_)].theta
			y_predtr =  my_ScalerY.de_norm(models2_4["deg." + str(i) + " : lambda_=" + str(lambd_)].predict_(X_tr_))
			mse2tr[j].insert(0, models2_4["deg." + str(i) + " : lambda_=" + str(lambd_)].mse_(y_train, y_predtr))
			rscore2tr[j].insert(0, models2_4["deg." + str(i) + " : lambda_=" + str(lambd_)].rscore_(y_train, y_predtr))
			print(f"R score = {rscore2tr[j][0]}")
			print(f"mse = {mse2tr[j][0]}")

		# save thetas for faster computing
		# with open("thetas_best.pickle","wb") as f:
		# 	pickle.dump(thetas, f)

		mse2 = [[], [], [], [], [], []]
		rscore2 = [[], [], [], [], [], []]
		colors = ['tab:purple', 'g', 'c', 'y', 'r', 'tab:pink']
		_, fig = plt.subplots(1, 3, figsize=(20, 8))
		labels = ['weight','prod_distance','time_delivery']
		for i in range(3):
			fig[i].scatter(x_test[:, i], y_test, marker='o', c='b', label="True price")
		for j in range(6):
			lambd_ = 2 * j / 10
			i = degree
			X_te_ = add_polynomial_features(X_te[:, :n_feature], degree)
			y_pred =  my_ScalerY.de_norm(models2_4["deg." + str(i) + " : lambda_=" + str(lambd_)].predict_(X_te_))
			mse2[j].insert(0, models2_4["deg." + str(i) + " : lambda_=" + str(lambd_)].mse_(y_test, y_pred))
			rscore2[j].insert(0, models2_4["deg." + str(i) + " : lambda_=" + str(lambd_)].rscore_(y_test, y_pred))
			for k in range(3):
				fig[k].scatter(x_test[:, k], y_pred, marker='x', c=colors[j], label="Prediction " + r"$\lambda$=" + str(lambd_))
				fig[k].legend()
				fig[k].set_xlabel(labels[k])
				fig[k].set_ylabel("Target")
		plt.grid()
		title = "Predictions of test data vs. true data on best model\n\n" + " "*10 + "R score" + " "*30 + "MSE\n" + "\n".join(r"$\lambda}$ = " + str(2 * j / 10) + " :  " + str(rscore2[j][0]) + "\t , \t" + str(mse2[j][0]) for j in range(6))
		plt.suptitle(title)
		plt.tight_layout()


		# # # load the other models from models.pickle. Then evaluate and plot the different graphics.
		with open("models.pickle", "rb") as f:
			models = pickle.load(f)

		mse2tr = [[], [], [], [], [], []]
		rscore2tr = [[], [], [], [], [], []]
		for j in range(6):
			lambd_ = 2 * j / 10
			for degree in range(4, 0, -1):
				i = degree
				X_tr_ = add_polynomial_features(X_tr[:, :n_feature], degree)
				y_predtr =  my_ScalerY.de_norm(models["deg." + str(i) + " : lambda_=" + str(lambd_)].predict_(X_tr_))
				mse2tr[j].insert(0, models["deg." + str(i) + " : lambda_=" + str(lambd_)].mse_(y_train, y_predtr))
				rscore2tr[j].insert(0, models["deg." + str(i) + " : lambda_=" + str(lambd_)].rscore_(y_train, y_predtr))

		# # # predict and plot data
		mse2 = [[], [], [], [], [], []]
		rscore2 = [[], [], [], [], [], []]
		for j in range(6):
			lambd_ = 2 * j / 10
			for i in range(4, 0, -1):
				degree = i
				X_te_ = add_polynomial_features(X_te[:, :n_feature], degree)
				y_pred =  my_ScalerY.de_norm(models["deg." + str(i) + " : lambda_=" + str(lambd_)].predict_(X_te_))
				mse2[j].insert(0, models["deg." + str(i) + " : lambda_=" + str(lambd_)].mse_(y_test, y_pred))
				rscore2[j].insert(0, models["deg." + str(i) + " : lambda_=" + str(lambd_)].rscore_(y_test, y_pred))


		# # # plot mse or score of determination

		tx = range(4)
		_, fig = plt.subplots(1,6, figsize=(40,10))
		for j in range(6):
			d = 2 * j / 10
			fig[j].set_xticks(tx)
			df = pd.DataFrame({'Training mse': rscore2tr[j], 'test mse': rscore2[j]}, index=tx)
			axes = df.plot.bar(ax=fig[j], rot=0, color={'Training mse': "orange", 'test mse': "blue"})
			axes.legend(loc=4)
			fig[j].set_xticklabels(["1","2","3","4"])
			fig[j].set_ylim([0, 1])
			fig[j].set_xlabel("degree")
			fig[j].set_ylabel("R score")
			fig[j].set_title(r"$\lambda}$ = " + str(d))
		title = "\n".join(r"$\lambda}$ = " + str(2 * j / 10) + " :  " + str(rscore2[j]) + "\t" + str(rscore2tr[j]) for j in range(6))
		plt.suptitle("Score of determination of the models vs. penalty term\n\n" + "test data R score                                                                             training data R score\n" + title)
		plt.subplots_adjust(top=0.75)


		tx = range(4)
		_, fig = plt.subplots(1,6, figsize=(40, 10))
		for j in range(6):
			d = 2 * j / 10
			fig[j].set_xticks(tx)
			df = pd.DataFrame({'Training mse': mse2tr[j], 'test mse': mse2[j]}, index=tx)
			axes = df.plot.bar(ax=fig[j], rot=0, color={'Training mse': "orange", 'test mse': "blue"})
			axes.legend(loc=1) 
			fig[j].set_xticklabels(["1","2","3","4"])
			fig[j].set_xlabel("degree")
			fig[j].set_ylabel("mse")
			fig[j].set_title(r"$\lambda}$ = " + str(d))
		title = "\n".join(r"$\lambda}$ = " + str(2 * j / 10) + " :  " + str(mse2[j]) + "\t" + str(mse2tr[j]) for j in range(6))
		plt.suptitle("MSE of the models vs. penalty term" + "\n\ntest data MSE                                                                             training data MSE\n" + title)
		plt.subplots_adjust(top=0.75)
		plt.show()

	except Exception as e:
		print(e)