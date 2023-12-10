import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from my_logistic_regression import MyLogisticRegression as MyLR
from data_splitter import data_splitter
from scaler import Standard_Scaler
import pickle
from metrics import Metrics

def relabel(y, fav_label):
	try:
		assert isinstance(
				y, np.ndarray) and (y.ndim == 1 or y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
		assert isinstance(fav_label, int) and fav_label in {0, 1, 2, 3}, "2nd argument must be a int that is either 0, 1 ,2 or 3"
		return(np.array([1 if yi==fav_label else 0 for yi in y])).reshape(-1, 1)

	except Exception as e:
		print(e)

def score_(y, y_pred):
	try:
		assert isinstance(
				y, np.ndarray) and (y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
		assert isinstance(
				y_pred, np.ndarray) and (y_pred.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
		return sum([1 if yi==yhi else 0 for yi, yhi in zip(y, y_pred)]) / y.shape[0]

	except Exception as e:
		print(e)


def scatter_plot(fig, x1, x2, y_test, y_pred, xlabel, ylabel):
	try:
		assert isinstance(
				x1, np.ndarray) and (x1.ndim == 1), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
		assert isinstance(
				x2, np.ndarray) and (x2.ndim == 1), "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
		assert isinstance(
				y_test, np.ndarray) and (y_test.ndim == 1), "3rd argument must be a numpy.ndarray, a vector of dimension m * 1"
		assert isinstance(
				y_pred, np.ndarray) and (y_pred.ndim == 1), "4th argument must be a numpy.ndarray, a vector of dimension m * 1"
		assert isinstance(xlabel, str) and isinstance(ylabel, str), "5th, 6th and 7th arguments must be strings"

		fig.scatter(x1[(y_test == 0)], x2[(y_test == 0)], s = 200, color='tab:gray', label="true values: Venus")
		fig.scatter(x1[(y_test == 1)], x2[(y_test == 1)], s = 200, color='c', label="true values: Earth")
		fig.scatter(x1[(y_test == 2)], x2[(y_test == 2)], s = 200, color='tab:pink', label="true values: Mars")
		fig.scatter(x1[(y_test == 3)], x2[(y_test == 3)], s = 200, color='y', label="true values: Asteroids\' Belt")
		fig.scatter(x1[(y_pred == 0)], x2[(y_pred == 0)], marker='x', color='g', label="predictions: Venus")
		fig.scatter(x1[(y_pred == 1)], x2[(y_pred == 1)], marker='x', color='b', label="predictions: Earth")
		fig.scatter(x1[(y_pred == 2)], x2[(y_pred == 2)], marker='x', color='tab:purple', label="predictions: Mars")
		fig.scatter(x1[(y_pred == 3)], x2[(y_pred == 3)], marker='x', color='tab:brown', label="predictions: Asteroids\' Belt")
		fig.set_xlabel(xlabel)
		fig.set_ylabel(ylabel)
		fig.grid()
		fig.legend()

	except Exception as e:
		print(e)

if __name__ == "__main__":
	try:
		# 1. Load data
		path = "solar_system_census.csv"
		data = pd.read_csv(path)
		x = data[['weight','height','bone_density']].values
		path = "solar_system_census_planets.csv"
		data = pd.read_csv(path)
		y = data[['Origin']].values
		planets = ["Venus", "Earth", "Mars", "Asteroids\' Belt"]

		# 2. split data
		(x_train, x_test_, y_train, y_test_) = data_splitter(x, y, 0.6)
		(x_test, x_cv, y_test, y_cv) = data_splitter(x_test_, y_test_, 0.5)
		
		# save split data into csv files
		columnsX = ["weight", "height", "bone_density"]
		columnsY = ["Origin"]
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
		# Zscore
		my_ScalerX = Standard_Scaler()
		my_ScalerX.fit(x_train)
		X_tr = my_ScalerX.transform(x_train)
		X_cv = my_ScalerX.transform(x_cv)

		# 4. add polynomial features & train models
		degree = 3
		X_tr_ = add_polynomial_features(X_tr, degree)
		X_cv_ = add_polynomial_features(X_cv, degree)

		models = {}
		y_tr_ = {}
		y_cv_ = {}
		y_pred_tr = {}
		y_pred_cv = {}
		f1_scores = {}
		for i in range(4):
			# 4.a relabel y labels
			y_tr_[i] = relabel(y_train, i)
			y_cv_[i] = relabel(y_cv, i)

		print("\n  ___________\n |           |\n | F1 scores |\n |___________|\n")
		for j in range(6):
			lambd_ = 2 * j / 10
			y_pred_tr_ = np.array([])
			y_pred_cv_ = np.array([])
			print(" * lambda_= " + str(lambd_) + " *")
			for i in range(4):
				# 4.b training
				models["class" + str(i) + " : lambda_=" + str(lambd_)] = MyLR(theta = np.ones((X_tr_.shape[1] + 1, 1)), alpha=5e-1, max_iter=2000, lambda_=lambd_)
				models["class" + str(i) + " : lambda_=" + str(lambd_)].fit_(X_tr_, y_tr_[i])
				
				# 5. Predict for each example the class according to each classifiers and select the one with the highest output probability.
				yp_tr = models["class" + str(i) + " : lambda_=" + str(lambd_)].predict_(X_tr_)
				if y_pred_tr_.any():
					y_pred_tr_ = np.hstack((y_pred_tr_, yp_tr))
				else:
					y_pred_tr_ = yp_tr

				yp_cv = models["class" + str(i) + " : lambda_=" + str(lambd_)].predict_(X_cv_)
				if y_pred_cv_.any():
					y_pred_cv_ = np.hstack((y_pred_cv_, yp_cv))
				else:
					y_pred_cv_ = yp_cv

				# 6. Evaluate f1_score
				f1_scores["cv - class" + str(i) + " : lambda_=" + str(lambd_)] = Metrics.f1_score_(y_cv_[i], yp_cv)
				print(planets[i] + " : ", f1_scores["cv - class" + str(i) + " : lambda_=" + str(lambd_)])
			print("\n")
			# 7. Calculate and display the fraction of correct predictions over the total number of predictions based on the test set and compare it to the train set.
			# print("y_pred_tr_ : ", y_pred_tr_)
			y_pred_tr[lambd_] = np.argmax(y_pred_tr_, axis=1).reshape(-1,1)
			y_pred_cv[lambd_] = np.argmax(y_pred_cv_, axis=1).reshape(-1,1)

		
		# 8. Plot scatter plots (one for each pair of citizen features) with the dataset and the final prediction of the model.
		for j in range(6):
			lambd_ = 2 * j / 10
			_, fig = plt.subplots(1, 3, figsize=(24, 10))
			labels = ['weight','height','bone_density']
			scatter_plot(fig[0], x_cv[:, 0], x_cv[:, 1], y_cv.reshape(-1,), y_pred_cv[lambd_].reshape(-1,), labels[0], labels[1])
			scatter_plot(fig[1], x_cv[:, 0], x_cv[:, 2], y_cv.reshape(-1,), y_pred_cv[lambd_].reshape(-1,), labels[0], labels[2])
			scatter_plot(fig[2], x_cv[:, 2], x_cv[:, 1], y_cv.reshape(-1,), y_pred_cv[lambd_].reshape(-1,), labels[2], labels[1])
			plt.suptitle("Scatter plots with the dataset and the final prediction of the model\n" \
				+ "fraction of correct predictions for cv data:  " +  str(score_(y_pred_cv[lambd_], y_cv)) + "\n"\
				+ "fraction of correct predictions for train data:  " +  str(score_(y_pred_tr[lambd_], y_train)) + "\n"\
				"lambda_=" + str(lambd_) + "\n"\
				+ "f1 score:" + "\n"\
				+ "Venus : " + str(f1_scores["cv - class" + str(0) + " : lambda_=" + str(lambd_)]) + "\n"\
				+ "Earth : " + str(f1_scores["cv - class" + str(1) + " : lambda_=" + str(lambd_)]) + "\n"\
				+ "Mars : " + str(f1_scores["cv - class" + str(2) + " : lambda_=" + str(lambd_)]) + "\n"\
				+ "Asteroids\' Belt : " + str(f1_scores["cv - class" + str(3) + " : lambda_=" + str(lambd_)]))
			plt.tight_layout() 


		for j in range(6):
			lambd_ = 2 * j / 10
			_, fig = plt.subplots(1, 3, figsize=(24, 10))
			labels = ['weight','height','bone_density']
			scatter_plot(fig[0], x_train[:, 0], x_train[:, 1], y_train.reshape(-1,), y_pred_tr[lambd_].reshape(-1,), labels[0], labels[1])
			scatter_plot(fig[1], x_train[:, 0], x_train[:, 2], y_train.reshape(-1,), y_pred_tr[lambd_].reshape(-1,), labels[0], labels[2])
			scatter_plot(fig[2], x_train[:, 2], x_train[:, 1], y_train.reshape(-1,), y_pred_tr[lambd_].reshape(-1,), labels[2], labels[1])
			plt.suptitle("Scatter plots with the train dataset and the final prediction of the model\n" \
				+ "fraction of correct predictions for cv data:  " +  str(score_(y_pred_cv[lambd_], y_cv)) + "\n"\
				+ "fraction of correct predictions for train data:  " +  str(score_(y_pred_tr[lambd_], y_train)) + "\n"\
				"lambda_=" + str(lambd_))

		# 9. Saving multiple models into pickle files 
		# https://stackoverflow.com/questions/42350635/how-can-you-pickle-multiple-objects-in-one-file

		with open("models.pickle","wb") as f:
			pickle.dump(models, f)

		# plt.show()

	except Exception as e:
		print(e)
