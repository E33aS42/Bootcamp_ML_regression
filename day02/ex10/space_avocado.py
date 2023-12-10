import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR
from data_spliter import data_spliter
from scaler import Minmax_Scaler, Standard_Scaler
import pickle
from benchmark_train import plot_pred


if __name__ == "__main__":
	try:
		# 1. Load data
		path = "space_avocado.csv"
		data = pd.read_csv(path)
		x = data[['weight','prod_distance','time_delivery']].values
		y = data[['target']].values

		# 2. split data
		(x_train, x_test, y_train, y_test) = data_spliter(x, y, 0.7)

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
		n_feature = 2

		####### 2 features #######
		# 'weight' & 'prod_distance'
		# We choose to ignore time_delivery as it looks like it is just noise data
		n_feature = 2
		models_best = {}

		# degree 3
		degree = 3
		X_3 = add_polynomial_features(X_tr[:, :n_feature], degree)
		models_best[3] = MyLR(theta = np.ones(1 + n_feature * degree).reshape(-1,1), alpha = 0.2, max_iter = 1000000)
		theta3 = np.array([[-1.41053674e-01], [ 9.38854751e-01], [-3.42396866e-01], [-1.23561340e-01], [ 2.62856113e-01], [-3.36546170e-04], [ 1.43515470e-01]])
		models_best[3] = MyLR(theta = theta3, alpha = 0.2, max_iter = 10000)
		models_best[3].fit_(X_3,Y_tr)
		print(models_best[3].theta)

		# degree 4
		degree = 4
		X_4 = add_polynomial_features(X_tr[:, :n_feature], degree)
		models_best[4] = MyLR(theta = np.ones(1 + n_feature * degree).reshape(-1,1), alpha = 0.002, max_iter = 1000000)
		theta4 = np.array([[-1.66050144e-01], [ 9.38545654e-01], [-3.52558437e-01], [-1.24096031e-01], [ 3.51070413e-01], [-1.06604191e-04], [ 1.49342809e-01], [ 1.62584829e-04], [-3.09964760e-02]])
		models_best[4] = MyLR(theta = theta4, alpha = 0.002, max_iter = 10000)
		models_best[4].fit_(X_4,Y_tr)
		# print(models_best[4].theta)

		# predict and plot data

		mse_best = {}
		rscore_best = {}

		# for i in [3, 4]:
		i = 4
		degree = i
		X_te_ = add_polynomial_features(X_te[:, :n_feature], degree)
		y_pred =  my_ScalerY.de_norm(models_best[degree].predict_(X_te_))
		mse_best["test data: deg. " + str(i)] = models_best[degree].mse_(y_test, y_pred)
		rscore_best["test data: deg. " + str(i)] = models_best[degree].rscore_(y_test, y_pred)
		
		plot_pred(models_best[i], x_test, y_test, y_pred, "Test data: 2 features - degree " + str(degree) + "\n MSE:" + str(np.round(mse_best["test data: deg. " + str(i)])) +  "\n R score:" + str(rscore_best["test data: deg. " + str(i)]))

		X_tr_ = add_polynomial_features(X_tr[:, :n_feature], degree)
		y_pred_tr =  my_ScalerY.de_norm(models_best[degree].predict_(X_tr_))
		mse_best["training data"] = models_best[degree].mse_(y_train, y_pred_tr)
		rscore_best["training data"] = models_best[degree].rscore_(y_train, y_pred_tr)
		plot_pred(models_best[i], x_train, y_train, y_pred_tr, "Training data: 2 features - degree " + str(degree) + "\n MSE:" + str(np.round(mse_best["training data"])) +  "\n R score:" + str(rscore_best["training data"]))

		print("mse_best: ", mse_best)
		print("rscore_best: ", rscore_best)

		# load the other models from models.pickle. Then evaluate and plot the different graphics.
		with open("models.pickle", "rb") as f:
			models3 = pickle.load(f)
			models2 = pickle.load(f)
			models10 = pickle.load(f)
			models11 = pickle.load(f)

		n_feature = 3
		mse3 = []
		rscore3 = []
		for i in range(4, 0, -1):
			degree = i
			X_te_ = add_polynomial_features(X_te[:, :n_feature], degree)
			y_pred =  my_ScalerY.de_norm(models3[degree].predict_(X_te_))
			mse3.insert(0, models3[degree].mse_(y_test, y_pred))
			rscore3.insert(0, models3[degree].rscore_(y_test, y_pred))
			plot_pred(models3[i], x_test, y_test, y_pred, "3 features - degree " + str(degree))

		n_feature = 2
		mse2 = []
		rscore2 = []
		for i in range(4, 0, -1):
			degree = i
			X_te_ = add_polynomial_features(X_te[:, :n_feature], degree)
			y_pred =  my_ScalerY.de_norm(models2[degree].predict_(X_te_))
			mse2.insert(0, models2[degree].mse_(y_test, y_pred))
			rscore2.insert(0, models2[degree].rscore_(y_test, y_pred))
			plot_pred(models2[i], x_test, y_test, y_pred, "2 features - degree " + str(degree))

		n_feature = 1
		feature = 0
		mse10 = []
		rscore10 = []
		for i in range(4, 0, -1):
			degree = i
			X_te_ = add_polynomial_features(X_te[:, feature], degree)
			y_pred =  my_ScalerY.de_norm(models10[degree].predict_(X_te_))
			mse10.insert(0, models10[degree].mse_(y_test, y_pred))
			rscore10.insert(0, models3[degree].rscore_(y_test, y_pred))
			plot_pred(models10[i], x_test, y_test, y_pred, "1 feature: weight - degree " + str(degree))

		n_feature = 1
		feature = 1
		mse11 = []
		rscore11 = []
		for i in range(4, 0, -1):
			degree = i
			X_te_ = add_polynomial_features(X_te[:, feature], degree)
			y_pred =  my_ScalerY.de_norm(models11[degree].predict_(X_te_))
			mse11.insert(0, models11[degree].mse_(y_test, y_pred))
			rscore11.insert(0, models3[degree].rscore_(y_test, y_pred))
			plot_pred(models11[i], x_test, y_test, y_pred, "1 feature: prod_distance - degree " + str(degree))

		# plot mse or score of determination
		import pandas as pd
		tx = range(1, 5)
		mse_list = [mse11, mse10, mse2, mse3]
		rscore_list = [rscore11, rscore10, rscore2, rscore3]
		titles = ["1 feature: weight", "1 feature: prod_distance", "2 features", "3 features"]
		_, fig = plt.subplots(1, 4, figsize=(24, 10))
		for i in range(4):
			# fig[i].bar(deg, mse_list[i], width = 0.4)
			fig[i].set_xticks(tx)
			fig[i].bar(tx, rscore_list[i], width = 0.4)
			fig[i].set_xticklabels(["1","2","3","4"])
			fig[i].set_ylim([0, 1])
			fig[i].set_xlabel("degree")
			fig[i].set_ylabel("R score")
			fig[i].set_title(titles[i])
		plt.suptitle("Score of determination of the models")
		plt.show()



	except Exception as e:
		print(e)