import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from my_logistic_regression import MyLogisticRegression as MyLR
from data_splitter import data_splitter
from scaler import Minmax_Scaler, Standard_Scaler


def relabel(y, fav_label):
    try:
        assert isinstance(
            y, np.ndarray) and (y.ndim == 1 or y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(fav_label, int) and fav_label in {
            0, 1, 2, 3}, "2nd argument must be a int that is either 0, 1 ,2 or 3"
        return(np.array([1 if yi == fav_label else 0 for yi in y])).reshape(-1, 1)

    except Exception as e:
        print(e)


def score_(y, y_pred):
    try:
        assert isinstance(
            y, np.ndarray) and (y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(
            y_pred, np.ndarray) and (y_pred.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        return sum([1 if yi == yhi else 0 for yi, yhi in zip(y, y_pred)]) / y.shape[0]

    except Exception as e:
        print(e)


def features_plot(fig, x, y_test, y_pred, title, xlabel):
    try:
        assert isinstance(
            x, np.ndarray) and (x.ndim == 1), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(
            y_test, np.ndarray) and (y_test.ndim == 1), "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(title, str), "3rd argument must be a string"

        fig.scatter(x, y_test, s=80, label="true values")
        fig.scatter(x, y_pred, c='r', marker='x',
                    linewidth=2, label="predictions")
        fig.set_xlabel(xlabel)
        fig.set_ylabel("Binarized Zipcode")
        fig.grid()
        fig.legend(loc='center right')

    except Exception as e:
        print(e)


def scatter_plot(fig, x1, x2, y_test, y_pred, xlabel, ylabel, title):
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

        fig.scatter(x1[(y_test == 0)], x2[(y_test == 0)],
                    color='b', label="Dataset")
        # fig.scatter(x1[(y_pred == 1)], x2[(y_pred == 1)], color='r', label=title)
        fig.scatter(x1[(y_test == 1)], x2[(y_test == 1)], s=200,
                    color='tab:pink', label="true values: " + title)
        fig.scatter(x1[(y_pred == 1)], x2[(y_pred == 1)], marker='x',
                    color='tab:purple', linewidth=2, label="predictions: " + title)
        fig.set_xlabel(xlabel)
        fig.set_ylabel(ylabel)
        fig.grid()
        fig.legend()

    except Exception as e:
        print(e)


if __name__ == "__main__":
    try:
        args = sys.argv
        assert len(args) == 2, "one zipcode argument is required"
        assert args[1].isdigit(
        ), "zipcode must be an integer either 0, 1, 2 or 3"
        zipcode = int(args[1])
        assert zipcode in {
            0, 1, 2, 3}, "zipcode argument must be an integer either 0, 1, 2 or 3"

        # 1. Load data
        path = "solar_system_census.csv"
        citizens_data = pd.read_csv(path)
        path = "solar_system_census_planets.csv"
        citizens_homeland = pd.read_csv(path)
        x = citizens_data[['weight', 'height', 'bone_density']].values
        y = citizens_homeland[['Origin']].values
        # print(x.shape, y.shape)

        # 2. relabel y labels
        fav_label = zipcode
        assert fav_label in {
            0, 1, 2, 3}, "zipcode argument must be an integer either 0, 1, 2 or 3"
        y = relabel(y, fav_label)
        # print(y)

        # 3. split data
        (x_train, x_test, y_train, y_test) = data_splitter(x, y, 0.7)
        # print(x_train.shape, y_train.shape,x_test.shape, y_test.shape)

        # 4. Normalization
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

        # 5. train a logistic model
        model = MyLR(np.ones((x.shape[1] + 1, 1)), alpha=5e-2, max_iter=100)
        model.fit_(X_tr, y_train)
        print(model.theta)

        # 6. calculate and display the fraction of correct predictions over the total number of predictions based on the test set
        y_pred = model.predict_(X_te)
        print("Model loss:\n", model.loss_(y_test, y_pred))
        print("Fraction of correct predictions: \n", score_(y_test, y_pred))

        title = ""
        if (fav_label == 0):
            title = "The flying cities of Venus"
        elif (fav_label == 1):
            title = "United Nations of Earth"
        elif (fav_label == 2):
            title = "Mars Republic"
        elif (fav_label == 3):
            title = "The Asteroids\' Belt colonies"

        # x_test = de_minmax(x_test, minx, maxx)
        # x_test = de_zscore(x_test, std, mu)
        _, fig = plt.subplots(1, 3, figsize=(20, 5))
        features_plot(fig[0], x_test[:, 0], y_test.reshape(-1,), y_pred.reshape(-1,),
                      "Logistic regression of normalized weight vs. binarized zipcode", 'weight')
        features_plot(fig[1], x_test[:, 1], y_test.reshape(-1,), y_pred.reshape(-1,),
                      "Logistic regression of normalized height vs. binarized zipcode", 'height')
        features_plot(fig[2], x_test[:, 2], y_test.reshape(-1,), y_pred.reshape(-1,),
                      "Logistic regression of normalized bone density vs. binarized zipcode", 'bone_density')
        plt.suptitle("Logistic regression - " + title)
        plt.show()

        # 7. Plot 3 scatter plots (one for each pair of citizen features) with the dataset and the final prediction of the model
        _, fig = plt.subplots(1, 3, figsize=(20, 10))
        labels = ['weight', 'height', 'bone_density']
        scatter_plot(fig[0], x_test[:, 0], x_test[:, 1], y_test.reshape(-1,),
                     y_pred.reshape(-1,), labels[0], labels[1], title)
        scatter_plot(fig[1], x_test[:, 0], x_test[:, 2], y_test.reshape(-1,),
                     y_pred.reshape(-1,), labels[0], labels[2], title)
        scatter_plot(fig[2], x_test[:, 2], x_test[:, 1], y_test.reshape(-1,),
                     y_pred.reshape(-1,), labels[2], labels[1], title)
        plt.suptitle("Scatter plots: dataset true values vs. final predictions of the model"
                     + "\nFraction of correct predictions: " + str(score_(y_test, y_pred)))
        plt.show()

    except Exception as e:
        print(e)
