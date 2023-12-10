import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from my_logistic_regression import MyLogisticRegression as MyLR
from data_splitter import data_splitter
from scaler import Minmax_Scaler, Standard_Scaler
from mono_log import relabel, score_


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
        assert isinstance(xlabel, str) and isinstance(
            ylabel, str), "5th, 6th and 7th arguments must be strings"

        fig.scatter(x1[(y_test == 0)], x2[(y_test == 0)], s=200,
                    color='c', label="true values: Venus")
        fig.scatter(x1[(y_test == 1)], x2[(y_test == 1)], s=200,
                    color='tab:pink', label="true values: Earth")
        fig.scatter(x1[(y_test == 2)], x2[(y_test == 2)], s=200,
                    color='tab:gray', label="true values: Mars")
        fig.scatter(x1[(y_test == 3)], x2[(y_test == 3)], s=200,
                    color='y', label="true values: Asteroids\' Belt")
        fig.scatter(x1[(y_pred == 0)], x2[(y_pred == 0)],
                    marker='x', color='b', label="predictions: Venus")
        fig.scatter(x1[(y_pred == 1)], x2[(y_pred == 1)], marker='x',
                    color='tab:purple', label="predictions: Earth")
        fig.scatter(x1[(y_pred == 2)], x2[(y_pred == 2)],
                    marker='x', color='g', label="predictions: Mars")
        fig.scatter(x1[(y_pred == 3)], x2[(y_pred == 3)], marker='x',
                    color='tab:brown', label="predictions: Asteroids\' Belt")
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
        citizens_data = pd.read_csv(path)
        path = "solar_system_census_planets.csv"
        citizens_homeland = pd.read_csv(path)
        x = citizens_data[['weight', 'height', 'bone_density']].values
        y = citizens_homeland[['Origin']].values
        features = ['weight', 'height', 'bone_density']
        planets = ["The flying cities of Venus", "United Nations of Earth",
                   "Mars Republic", "The Asteroids\' Belt colonies"]

        # 2. split data
        (x_train, x_test, y_train, y_test) = data_splitter(x, y, 0.7)
        # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        # 3. Normalization
        # Minmax
        # myScaler = Minmax_Scaler()
        # myScaler.fit(x_train)
        # X_tr = myScaler.transform(x_train)
        # X_te = myScaler.transform(x_test)

        # Zscore
        my_Scaler = Standard_Scaler()
        my_Scaler.fit(x_train)
        X_tr = my_Scaler.transform(x_train)
        X_te = my_Scaler.transform(x_test)

        # 4. Training
        # We are going to train 4 logistic regression classifiers to discriminate each class from the others

        models = {}
        y_ = {}
        for i in range(4):
            # 4.a relabel y labels
            y_[i] = relabel(y_train, i)

            # 4.b training
            models[i] = MyLR(np.ones((X_tr.shape[1] + 1, 1)),
                             alpha=5e-2, max_iter=100)
            models[i].fit_(X_tr, y_[i])

        # 5. Predict for each example the class according to each classifiers and select the one with the highest output probability.
        y_pred_ = np.array([])
        for i in range(4):
            if y_pred_.any():
                y_pred_ = np.hstack((y_pred_, models[i].predict_(X_te)))
            else:
                y_pred_ = models[i].predict_(X_te)
            # print(y_pred_.shape, models[i].predict_(X_te).shape)

        y_pred_tr_ = np.array([])
        for i in range(4):
            if y_pred_tr_.any():
                y_pred_tr_ = np.hstack((y_pred_tr_, models[i].predict_(X_tr)))
            else:
                y_pred_tr_ = models[i].predict_(X_tr)

        # 6. Calculate and display the fraction of correct predictions over the total number of predictions based on the test set and compare it to the train set.
        y_pred = np.argmax(y_pred_, axis=1).reshape(-1, 1)
        print("fraction of correct predictions for test data:  ",
              score_(y_pred, y_test))
        y_pred_tr = np.argmax(y_pred_tr_, axis=1).reshape(-1, 1)
        print("fraction of correct predictions for train data:  ",
              score_(y_pred_tr, y_train))

        # 7. Plot 3 scatter plots (one for each pair of citizen features) with the dataset and the final prediction of the model.
        _, fig = plt.subplots(1, 3, figsize=(24, 10))
        labels = ['weight', 'height', 'bone_density']
        scatter_plot(fig[0], x_test[:, 0], x_test[:, 1],
                     y_test.reshape(-1,), y_pred.reshape(-1,), labels[0], labels[1])
        scatter_plot(fig[1], x_test[:, 0], x_test[:, 2],
                     y_test.reshape(-1,), y_pred.reshape(-1,), labels[0], labels[2])
        scatter_plot(fig[2], x_test[:, 2], x_test[:, 1],
                     y_test.reshape(-1,), y_pred.reshape(-1,), labels[2], labels[1])
        plt.suptitle("Scatter plots with the dataset and the final prediction of the model\n"
                     + "fraction of correct predictions for test data:  " +
                     str(score_(y_pred, y_test)) + "\n"
                     + "fraction of correct predictions for train data:  " + str(score_(y_pred_tr, y_train)))
        plt.show()

    except Exception as e:
        print(e)
