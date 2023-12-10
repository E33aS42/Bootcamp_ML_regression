import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR
from data_spliter import data_spliter
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
        _, fig = plt.subplots(1, 3, figsize=(24, 10))
        labels = ['weight', 'prod_distance', 'time_delivery']
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
        x = data[['weight', 'prod_distance', 'time_delivery']].values
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
        n_feature = 3

        # 5. add polynomial features & train models

        # # all 3 features
        models3 = {}
        # # degree 1
        models3[1] = MyLR(theta=np.array(
            [[1.0], [1.0], [1.0], [1.0]]), alpha=0.2, max_iter=10000)
        theta1 = np.array([[7.25898825e-16], [9.43694429e-01],
                          [-4.29262778e-02], [-5.02815930e-03]])
        models3[1] = MyLR(theta=theta1, alpha=0.2, max_iter=10000)
        models3[1].fit_(X_tr[:, :n_feature], Y_tr)
        # print(models3[1].theta)

        # # define degree
        degree = 2

        # # degree 2
        X_2 = add_polynomial_features(X_tr[:, :n_feature], degree)
        models3[2] = MyLR(theta=np.ones(
            1 + n_feature * degree).reshape(-1, 1), alpha=0.2, max_iter=1000000)
        theta2 = np.array([[-0.14755968], [0.94416075], [-0.05174469],
                          [-0.00437819], [-0.12465008], [0.27589419], [-0.00368443]])
        models3[2] = MyLR(theta=theta2, alpha=0.2, max_iter=10000)
        models3[2].fit_(X_2, Y_tr)
        # print(models3[2].theta)

        # # degree 3
        degree = 3
        X_3 = add_polynomial_features(X_tr[:, :n_feature], degree)
        models3[3] = MyLR(theta=np.ones(
            1 + n_feature * degree).reshape(-1, 1), alpha=0.2, max_iter=1000000)
        theta3 = np.array([[-1.43572153e-01], [9.44850412e-01], [-3.48831236e-01], [-5.52008859e-03], [-1.24545231e-01],
                          [2.65826855e-01], [-1.41989187e-03], [6.38470097e-04], [1.46708439e-01], [-2.49974177e-05]])
        models3[3] = MyLR(theta=theta3, alpha=0.2, max_iter=10000)
        models3[3].fit_(X_3, Y_tr)
        # print(models3[3].theta)

        # degree 4
        degree = 4
        X_4 = add_polynomial_features(X_tr[:, :n_feature], degree)
        models3[4] = MyLR(theta=np.ones(
            1 + n_feature * degree).reshape(-1, 1), alpha=0.002, max_iter=100000)
        theta4 = np.array([[-1.61260237e-01], [9.45368949e-01], [-3.51502407e-01], [-5.58773174e-03], [-1.24447246e-01], [3.50510094e-01],
                          [-1.46438143e-03], [-9.65084955e-06], [1.48353454e-01], [1.29988243e-05], [-2.13494090e-05], [-3.08344531e-02], [-2.07170329e-05]])
        models3[4] = MyLR(theta=theta4, alpha=0.002, max_iter=10000)
        models3[4].fit_(X_4, Y_tr)
        # print(models3[4].theta)

        # predict data
        mse3 = []
        rscore3 = []
        for i in range(4, 0, -1):
            degree = i
            X_te_ = add_polynomial_features(X_te, degree)
            y_pred = my_ScalerY.de_norm(models3[degree].predict_(X_te_))
            mse3.insert(0, models3[degree].mse_(y_test, y_pred))
            rscore3.insert(0, models3[degree].rscore_(y_test, y_pred))
            plot_pred(models3[i], x_test, y_test, y_pred,
                      "3 features - degree " + str(degree))

        ####### 2 features #######
        # 'weight' & 'prod_distance'
        # We choose to ignore time_delivery as it looks like it is just noise data
        n_feature = 2
        models2 = {}

        # degree 1
        models2[1] = MyLR(theta=np.array(
            [[1.0], [1.0], [1.0]]), alpha=0.2, max_iter=10000)
        theta1 = np.array(
            [[-6.10330096e-16], [9.42431431e-01], [-4.05324970e-02]])
        models2[1] = MyLR(theta=theta1, alpha=0.2, max_iter=10000)
        models2[1].fit_(X_tr[:, :n_feature], Y_tr)
        # print(models2[1].theta)

        # define degree
        degree = 2

        # degree 2
        X_2 = add_polynomial_features(X_tr[:, :n_feature], degree)
        models2[2] = MyLR(theta=np.ones(
            1 + n_feature * degree).reshape(-1, 1), alpha=0.2, max_iter=1000000)
        theta2 = np.array([[-0.14570967], [0.94532118],
                          [-0.05878055], [-0.12270099], [0.26841066]])
        models2[2] = MyLR(theta=theta2, alpha=0.2, max_iter=10000)
        models2[2].fit_(X_2, Y_tr)
        # print(models2[2].theta)

        # # degree 3
        degree = 3
        X_3 = add_polynomial_features(X_tr[:, :n_feature], degree)
        models2[3] = MyLR(theta=np.ones(
            1 + n_feature * degree).reshape(-1, 1), alpha=0.2, max_iter=1000000)
        theta3 = np.array([[-1.34655106e-01], [9.44468141e-01], [-3.53122023e-01],
                          [-1.24323293e-01], [2.59090039e-01], [8.70442150e-05], [1.46055912e-01]])
        models2[3] = MyLR(theta=theta3, alpha=0.2, max_iter=10000)
        models2[3].fit_(X_3, Y_tr)
        # print(models2[3].theta)

        # degree 4
        degree = 4
        X_4 = add_polynomial_features(X_tr[:, :n_feature], degree)
        models2[4] = MyLR(theta=np.ones(
            1 + n_feature * degree).reshape(-1, 1), alpha=0.002, max_iter=1000000)
        theta4 = np.array([[-1.59516760e-01], [9.45036786e-01], [-3.64349282e-01], [-1.24417127e-01], [
                          3.47599138e-01], [-3.18713086e-05], [1.52635707e-01], [-5.27419828e-05], [-3.14882517e-02]])
        models2[4] = MyLR(theta=theta4, alpha=0.002, max_iter=10000)
        models2[4].fit_(X_4, Y_tr)
        # print(models2[4].theta)

        # predict and plot data

        mse2 = []
        rscore2 = []

        for i in range(4, 0, -1):
            degree = i
            X_te_ = add_polynomial_features(X_te[:, :n_feature], degree)
            y_pred = my_ScalerY.de_norm(models2[degree].predict_(X_te_))
            mse2.insert(0, models2[degree].mse_(y_test, y_pred))
            rscore2.insert(0, models2[degree].rscore_(y_test, y_pred))
            plot_pred(models2[i], x_test, y_test, y_pred,
                      "2 features - degree " + str(degree))

        ######### 1 feature: 'weight' #########

        ######## 1 feature: 'prod_distance' ########
        n_feature = 1
        feature = 1
        models11 = {}
        # # degree 1
        models11[1] = MyLR(theta=np.array([[1.0], [1.0]]),
                           alpha=0.2, max_iter=10000)
        theta1 = np.array([[-6.17591096e-16], [-4.58682861e-02]])
        models11[1] = MyLR(theta=theta1, alpha=0.2, max_iter=10000)
        models11[1].fit_(X_tr[:, feature], Y_tr)
        print(models11[1].theta)

        # define degree
        degree = 2

        # degree 2
        X_2 = add_polynomial_features(X_tr[:, feature], degree)
        models11[2] = MyLR(theta=np.ones(
            1 + n_feature * degree).reshape(-1, 1), alpha=0.2, max_iter=1000000)
        theta2 = np.array([[-0.27827461], [-0.05317222], [0.27827461]])
        models11[2] = MyLR(theta=theta2, alpha=0.2, max_iter=10000)
        models11[2].fit_(X_2, Y_tr)
        # print(models11[2].theta)

        # # degree 3
        degree = 3
        X_3 = add_polynomial_features(X_tr[:, feature], degree)
        models11[3] = MyLR(theta=np.ones(
            1 + n_feature * degree).reshape(-1, 1), alpha=0.2, max_iter=1000000)
        theta3 = np.array([[-0.27190275], [-0.34374514],
                          [0.27000593], [0.14513104]])
        models11[3] = MyLR(theta=theta3, alpha=0.2, max_iter=10000)
        models11[3].fit_(X_3, Y_tr)
        # print(models11[3].theta)

        # degree 4
        degree = 4
        X_4 = add_polynomial_features(X_tr[:, feature], degree)
        models11[4] = MyLR(theta=np.ones(
            1 + n_feature * degree).reshape(-1, 1), alpha=0.002, max_iter=1000000)
        theta4 = np.array([[-0.30190266], [-0.35645905],
                          [0.37561684], [0.15260854], [-0.03779982]])
        models11[4] = MyLR(theta=theta4, alpha=0.002, max_iter=10000)
        models11[4].fit_(X_4, Y_tr)
        # print(models11[4].theta)

        # predict and plot data

        mse11 = []
        rscore11 = []
        for i in range(4, 0, -1):
            degree = i
            X_te_ = add_polynomial_features(X_te[:, feature], degree)
            y_pred = my_ScalerY.de_norm(models11[degree].predict_(X_te_))
            mse11.insert(0, models11[degree].mse_(y_test, y_pred))
            rscore11.insert(0, models11[degree].rscore_(y_test, y_pred))
            plot_pred(models11[i], x_test, y_test, y_pred,
                      "1 feature: prod_distance - degree " + str(degree))
        # plt.show()

        n_feature = 1
        feature = 0
        models10 = {}

        # degree 1
        models10[1] = MyLR(theta=np.array([[1.0], [1.0]]),
                           alpha=0.2, max_iter=10000)
        theta1 = np.array([[-3.39324817e-16], [9.43035828e-01]])
        models10[1] = MyLR(theta=theta1, alpha=0.2, max_iter=10000)
        models10[1].fit_(X_tr[:, feature], Y_tr)
        # print(models10[1].theta)

        # define degree
        degree = 2

        # degree 2
        X_2 = add_polynomial_features(X_tr[:, feature], degree)
        models10[2] = MyLR(theta=np.ones(
            1 + n_feature * degree).reshape(-1, 1), alpha=0.2, max_iter=1000000)
        theta2 = np.array([[0.12246541], [0.94465631], [-0.12246541]])
        models10[2] = MyLR(theta=theta2, alpha=0.2, max_iter=10000)
        models10[2].fit_(X_2, Y_tr)
        # print(models10[2].theta)

        # # degree 3
        degree = 3
        X_3 = add_polynomial_features(X_tr[:, feature], degree)
        models10[3] = MyLR(theta=np.ones(
            1 + n_feature * degree).reshape(-1, 1), alpha=0.2, max_iter=1000000)
        theta3 = np.array([[0.12246606], [0.94819817],
                          [-0.12243982], [-0.00198356]])
        models10[3] = MyLR(theta=theta3, alpha=0.2, max_iter=10000)
        models10[3].fit_(X_3, Y_tr)
        # print(models10[3].theta)

        # degree 4
        degree = 4
        X_4 = add_polynomial_features(X_tr[:, feature], degree)
        models10[4] = MyLR(theta=np.ones(
            1 + n_feature * degree).reshape(-1, 1), alpha=0.002, max_iter=1000000)
        theta4 = np.array([[0.12553517], [0.94836473],
                          [-0.13313963], [-0.00211003], [0.00427399]])
        models10[4] = MyLR(theta=theta4, alpha=0.002, max_iter=10000)
        models10[4].fit_(X_4, Y_tr)
        # print(models10[4].theta)

        # predict and plot data

        mse10 = []
        rscore10 = []
        for i in range(4, 0, -1):
            degree = i
            X_te_ = add_polynomial_features(X_te[:, feature], degree)
            y_pred = my_ScalerY.de_norm(models10[degree].predict_(X_te_))
            mse10.insert(0, models10[degree].mse_(y_test, y_pred))
            rscore10.insert(0, models10[degree].rscore_(y_test, y_pred))
            plot_pred(models10[i], x_test, y_test, y_pred,
                      "1 feature: weight - degree " + str(degree))

        # print(mse3)
        # print(mse2)
        # print(mse10)
        # print(mse11)

        # print(rscore3)
        # print(rscore2)
        # print(rscore10)
        # print(rscore11)

        # plot mse or score of determination
        import pandas as pd
        tx = range(1, 5)
        mse_list = [mse11, mse10, mse2, mse3]
        rscore_list = [rscore11, rscore10, rscore2, rscore3]
        titles = ["1 feature: prod_distance",
                  "1 feature: weight", "2 features", "3 features"]
        _, fig = plt.subplots(1, 4, figsize=(24, 10))
        for i in range(4):
            # fig[i].bar(deg, mse_list[i], width = 0.4)
            fig[i].set_xticks(tx)
            fig[i].bar(tx, rscore_list[i], width=0.4)
            fig[i].set_xticklabels(["1", "2", "3", "4"])
            fig[i].set_ylim([0, 1])
            fig[i].set_xlabel("degree")
            fig[i].set_ylabel("R score")
            fig[i].set_title(titles[i])
        plt.suptitle("Score of determination of the models")
        # plt.show()

        # 7. Saving multiple models into pickle files
        # https://stackoverflow.com/questions/42350635/how-can-you-pickle-multiple-objects-in-one-file

        with open("models.pickle", "wb") as f:
            pickle.dump(models3, f)
            pickle.dump(models2, f)
            pickle.dump(models10, f)
            pickle.dump(models11, f)

        # with open("models.pickle", "rb") as f:
        # 	pickle3 = pickle.load(f)
        # 	pickle2 = pickle.load(f)
        # 	pickle10 = pickle.load(f)
        # 	pickle11 = pickle.load(f)

        # n_feature = 3
        # for i in range(4):
        # 	degree = i + 1
        # 	X_te_ = add_polynomial_features(X_te[:, :n_feature], degree)
        # 	y_pred =  my_ScalerY.de_norm(pickle3[degree].predict_(X_te_))
        # 	plot_pred(pickle3[i + 1], x_test, y_test, y_pred, "3 features - degree " + str(degree))
        plt.show()

    except Exception as e:
        print(e)
