import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from my_logistic_regression import MyLogisticRegression as MyLR
from data_splitter import data_splitter
from scaler import Standard_Scaler
import pickle
from metrics import Metrics
from benchmark_train import relabel, score_, scatter_plot


def bar_plot(fig, data, total_width=0.8, single_width=1):
	"""Draws a bar plot with multiple bars per data point"""
	try: 
		# define default colors
		# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
		colors = ['g', 'b', 'r', 'y']

		# Number of bars per group
		n_bars = len(data)

		# The width of a single bar
		bar_width = total_width / n_bars

		# List containing handles for the drawn bars, used for the legend
		bars = []

		# Iterate over all data
		for i, (name, values) in enumerate(data.items()):
			# The offset in x direction of that bar
			x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

			# Draw a bar for every value of that type
			for x, y in enumerate(values):
				bar = fig.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i])

			# Add a handle to the last drawn bar, which we'll need for the legend
			bars.append(bar[0])

		# Draw legend
		fig.legend(bars, data.keys(), loc=7)

	except Exception as e:
		print(e)

def load_data(path, features):
	""" Load dataset from .csv file into numpy array"""
	try:
		data = pd.read_csv(path)
		return data[features].values

	except Exception as e:
		print(e)
		return None

if __name__ == "__main__":
	try:
		# 1. Load data
		planets = ["Venus", "Earth", "Mars", "Asteroids\' Belt"]
		x_features = ['weight','height','bone_density']
		y_feature = ['Origin']
		x_train = load_data("x_train.csv", x_features)
		x_test = load_data("x_test.csv", x_features)
		y_train = load_data("y_train.csv", y_feature)
		y_test = load_data("y_test.csv", y_feature)

		# 2. Load models
		with open("models.pickle", "rb") as f:
			models = pickle.load(f)

		# 3. Normalization
		# Zscore
		my_ScalerX = Standard_Scaler()
		my_ScalerX.fit(x_train)
		X_tr = my_ScalerX.transform(x_train)
		X_te = my_ScalerX.transform(x_test)

		# 4. add polynomial features & train models
		degree = 3
		X_tr_ = add_polynomial_features(X_tr, degree)
		X_te_ = add_polynomial_features(X_te, degree)

		models_te = {}
		y_tr_ = {}
		y_te_ = {}
		y_pred_tr = {}
		y_pred_te = {}
		f1_scores = {}
		for i in range(4):
			# 4.a relabel y labels
			y_tr_[i] = relabel(y_train, i)
			y_te_[i] = relabel(y_test, i)

		# Train with chosen model:
		lambd_ = 0.8
		y_pred_tr_ = np.array([])
		y_pred_te_ = np.array([])
		print(" * lambda_= " + str(lambd_) + " *")
		for i in range(4):
			# 4.b training
			models_te["class" + str(i) + " : lambda_=" + str(lambd_)] = MyLR(theta = np.ones((X_tr_.shape[1] + 1, 1)), alpha=5e-1, max_iter=2000, lambda_=lambd_)
			models_te["class" + str(i) + " : lambda_=" + str(lambd_)].fit_(X_tr_, y_tr_[i])
			
			# 4.c Predict for each example the class according to each classifiers and select the one with the highest output probability.
			yp_tr = models_te["class" + str(i) + " : lambda_=" + str(lambd_)].predict_(X_tr_)
			if y_pred_tr_.any():
				y_pred_tr_ = np.hstack((y_pred_tr_, yp_tr))
			else:
				y_pred_tr_ = yp_tr

			yp_te = models_te["class" + str(i) + " : lambda_=" + str(lambd_)].predict_(X_te_)
			if y_pred_te_.any():
				y_pred_te_ = np.hstack((y_pred_te_, yp_te))
			else:
				y_pred_te_ = yp_te

			# 4.d Evaluate f1_score
			f1_scores["te - class" + str(i) + " : lambda_=" + str(lambd_)] = Metrics.f1_score_(y_te_[i], yp_te)
			print(planets[i] + " : ", f1_scores["te - class" + str(i) + " : lambda_=" + str(lambd_)])
		print("\n")


		# 5. predict test data with all models:
		for j in range(6):
			lambd_ = 2 * j / 10
			y_pred_tr_ = np.array([])
			y_pred_te_ = np.array([])
			print(" * lambda_= " + str(lambd_) + " *")
			for i in range(4):
				# 5.a prediction on the test set

				yp_te = models["class" + str(i) + " : lambda_=" + str(lambd_)].predict_(X_te_)
				if y_pred_te_.any():
					y_pred_te_ = np.hstack((y_pred_te_, yp_te))
				else:
					y_pred_te_ = yp_te

				# 5.b Evaluate f1_score
				f1_scores["te - class" + str(i) + " : lambda_=" + str(lambd_)] = Metrics.f1_score_(y_te_[i], yp_te)
				print(planets[i] + " : ", f1_scores["te - class" + str(i) + " : lambda_=" + str(lambd_)])
			print("\n")
			# 5.c Calculate and display the fraction of correct predictions over the total number of predictions based on the test set:
			y_pred_te[lambd_] = np.argmax(y_pred_te_, axis=1).reshape(-1,1)
		for j in range(6):
			lambd_ = 2 * j / 10
			print(score_(y_pred_te[lambd_], y_test))
		# 6. bar plot of the f1 score for all models
		f1score_dict = {}
		for i in range(4):
			f1score_dict[planets[i]] = []
			for j in range(6):
				lambd_ = 2 * j / 10
				f1score_dict[planets[i]].append(f1_scores["te - class" + str(i) + " : lambda_=" + str(lambd_)])

		tx = range(0, 6)
		_, fig = plt.subplots()
		fig.set_xticks(tx)
		bar_plot(fig, f1score_dict)
		fig.set_xticklabels(["0.0","0.2","0.4","0.6","0.8","1.0"])
		fig.set_ylim([0, 1])
		fig.set_xlabel("lambda")
		fig.set_ylabel("f1 score")
		plt.suptitle("F1 scores of the models")
		plt.show()

		# 7. Visualize the target values and the predicted values of the model on the same scatterplot.
		# for j in range(6):
			# lambd_ = 2 * j / 10
		lambd_ = 0.6
		_, fig = plt.subplots(1, 3, figsize=(24, 10))
		labels = ['weight','height','bone_density']
		scatter_plot(fig[0], x_test[:, 0], x_test[:, 1], y_test.reshape(-1,), y_pred_te[lambd_].reshape(-1,), labels[0], labels[1])
		scatter_plot(fig[1], x_test[:, 0], x_test[:, 2], y_test.reshape(-1,), y_pred_te[lambd_].reshape(-1,), labels[0], labels[2])
		scatter_plot(fig[2], x_test[:, 2], x_test[:, 1], y_test.reshape(-1,), y_pred_te[lambd_].reshape(-1,), labels[2], labels[1])
		plt.suptitle("Scatter plots with the dataset and the final prediction of the model\n" \
			+ "fraction of correct predictions for test data:  " +  str(score_(y_pred_te[lambd_], y_test)) + "\n"\
			"lambda_=" + str(lambd_) + "\n"\
			+ "f1 score:" + "\n"\
			+ "Venus : " + str(f1_scores["te - class" + str(0) + " : lambda_=" + str(lambd_)]) + "\n"\
			+ "Earth : " + str(f1_scores["te - class" + str(1) + " : lambda_=" + str(lambd_)]) + "\n"\
			+ "Mars : " + str(f1_scores["te - class" + str(2) + " : lambda_=" + str(lambd_)]) + "\n"\
			+ "Asteroids\' Belt : " + str(f1_scores["te - class" + str(3) + " : lambda_=" + str(lambd_)]))
		plt.tight_layout() 

		plt.show()

	except Exception as e:
		print(e)