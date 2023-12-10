import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR


if __name__ == "__main__":
    try:
        path = "are_blue_pills_magics.csv"
        data = pd.read_csv(path)
        x = data['Micrograms'].values
        y = data['Score'].values

# Trains six separate Linear Regression models with polynomial hypothesis with degrees ranging from 1 to 6

        # my_lr_1 = MyLR(np.ones(2).reshape(-1,1), alpha=0.001, max_iter=1000)
        # my_lr_2 = MyLR(np.ones(3).reshape(-1,1), alpha=0.001, max_iter=1000)
        # my_lr_3 = MyLR(np.ones(4).reshape(-1,1), alpha=6.e-5, max_iter=10000)
        # theta3 = np.array([[ 54.79378422], [ 30.28001334], [-12.11078382], [1.08117902]])
        # my_lr_3 = MyLR(theta3, alpha=6.e-5, max_iter=1000)
        # my_lr_4 = MyLR(np.ones(5).reshape(-1,1), alpha=2.e-6, max_iter=1000)
        # theta4 = np.array([-20., 160., -80., 10., -1.]).reshape(-1, 1)
        # theta4 = np.array([[-20.39553254], [159.87414903], [-78.90651704], [ 14.36731696], [ -0.89812836]])
        # my_lr_4 = MyLR(theta4, alpha=2.e-6, max_iter=1000)
        # my_lr_5 = MyLR(np.ones(6).reshape(-1,1), alpha=1.e-8, max_iter=4000)
        # theta5 = np.array([1140., -1850., 1110., -305., 40., -2.]).reshape(-1, 1)
        # my_lr_5 = MyLR(theta5, alpha=1.e-8, max_iter=40000)
        # my_lr_6 = MyLR(np.ones(7).reshape(-1,1), alpha=1.e-9, max_iter=5000)
        # theta6 = np.array([9110., -18015., 13400., -4935., 966., -96.4, 3.86]).reshape(-1, 1)
        # my_lr_6 = MyLR(theta6, alpha=1.e-9, max_iter=50000)

        print("\n*** Training ***\n")
        fig = plt.figure()
        continuous_x = np.arange(1, 7.01, 0.01).reshape(-1, 1)

        # thetas_1 = np.array([[89.10512305], [-9.00574709]])
        my_lr_1 = MyLR(np.ones(2).reshape(-1, 1), alpha=0.05, max_iter=1000)
        # my_lr_1 = MyLR(thetas_1, alpha=0.05, max_iter=1000)
        thetas_1 = my_lr_1.fit_(x, y)
        print("thetas_1: \n", thetas_1)
        y_hat_1 = my_lr_1.predict_(x)
        mse_1 = my_lr_1.mse_(y, y_hat_1)
        plt.plot(x, y_hat_1, c='r', label="degree 1")

        # thetas_2 = np.array([[ 90.49715322], [-10.04965781], [0.14064899]])
        my_lr_2 = MyLR(np.ones(3).reshape(-1, 1), alpha=0.0028, max_iter=50000)
        # my_lr_2 = MyLR(thetas_2, alpha=0.0028, max_iter=1000)
        x_2 = add_polynomial_features(x, 2)
        thetas_2 = my_lr_2.fit_(x_2, y)
        print("thetas_2: \n", thetas_2)
        x2_ = add_polynomial_features(continuous_x, 2)
        y_hat_2 = my_lr_2.predict_(x2_)
        mse_2 = my_lr_2.mse_(y, my_lr_2.predict_(x_2))
        plt.plot(continuous_x, y_hat_2, c='g', label="degree 2")

        # theta3 = np.array([[ 52.34809151], [ 32.86974034], [-12.86795266], [  1.14639649]])
        # theta3 = np.array([[ 57.02385045], [ 27.86688249], [-11.39718053], [  1.01929858]])
        # theta3 = np.array([[ 60.98306914], [ 23.57990303], [-10.12902892], [  0.90930885]])
        # theta3 = np.array([[64.35889535], [19.92457465], [-9.04772334], [ 0.81552441]])
        theta3 = np.array([[75.08999518], [8.30499128],
                          [-5.61046161], [0.5174018]])
        my_lr_3 = MyLR(theta3, alpha=8.e-5, max_iter=1000)
        # my_lr_3 = MyLR(np.ones(4).reshape(-1,1), alpha=8.e-5, max_iter=5000000)
        x_3 = add_polynomial_features(x, 3)
        thetas_3 = my_lr_3.fit_(x_3, y)
        print("thetas_3: \n", thetas_3)
        x3_ = add_polynomial_features(continuous_x, 3)
        y_hat_3 = my_lr_3.predict_(x3_)
        mse_3 = my_lr_3.mse_(y, my_lr_3.predict_(x_3))
        plt.plot(continuous_x, y_hat_3, "--", c='b', label="degree 3")

        theta4 = np.array([[-20], [160], [-80], [10], [-1]]).reshape(-1, 1)
        my_lr_4 = MyLR(theta4, alpha=2.e-6, max_iter=10000)
        x_4 = add_polynomial_features(x, 4)
        thetas_4 = my_lr_4.fit_(x_4, y)
        print("thetas_4: \n", thetas_4)
        x4_ = add_polynomial_features(continuous_x, 4)
        y_hat_4 = my_lr_4.predict_(x4_)
        mse_4 = my_lr_4.mse_(y, my_lr_4.predict_(x_4))
        plt.plot(continuous_x, y_hat_4, c='y', label="degree 4")

        theta5 = np.array([[1140], [-1850], [1110], [-305],
                          [40], [-2]]).reshape(-1, 1)
        my_lr_5 = MyLR(theta5, alpha=1.e-8, max_iter=100000)
        x_5 = add_polynomial_features(x, 5)
        thetas_5 = my_lr_5.fit_(x_5, y)
        print("thetas_5: \n", thetas_5)
        x5_ = add_polynomial_features(continuous_x, 5)
        y_hat_5 = my_lr_5.predict_(x5_)
        mse_5 = my_lr_5.mse_(y, my_lr_5.predict_(x_5))
        plt.plot(continuous_x, y_hat_5, ':', c='k', label="degree 5")

        theta6 = np.array([[9110], [-18015], [13400], [-4935],
                          [966], [-96.4], [3.86]]).reshape(-1, 1)
        my_lr_6 = MyLR(theta6, alpha=1.e-9, max_iter=100000)
        x_6 = add_polynomial_features(x, 6)
        thetas_6 = my_lr_6.fit_(x_6, y)
        print("thetas_6: \n", thetas_6)
        print("fit done")
        x6_ = add_polynomial_features(continuous_x, 6)
        y_hat_6 = my_lr_6.predict_(x6_)
        mse_6 = my_lr_6.mse_(y, my_lr_6.predict_(x_6))
        plt.plot(continuous_x, y_hat_6, c='c', label="degree 6")

        plt.scatter(x, y)
        plt.xlabel("Micrograms")
        plt.ylabel("Score")
        plt.grid()
        plt.legend()
        plt.title("Polynomial regressions")
        plt.draw()
        print(
            f"\n*** MSE ***\n\ndegree 1: {mse_1}\ndegree 2: {mse_2}\ndegree 3: {mse_3}\ndegree 4: {mse_4}\ndegree 5: {mse_5}\ndegree 6: {mse_6}")

# Plots a bar plot showing the MSE score of the models in function of the polynomial degree of the hypothesis
        import seaborn
        import pandas as pd
        fig = plt.figure()
        deg = range(1, 7)
        mse_list = [mse_1, mse_2, mse_3, mse_4, mse_5, mse_6]
        plt.bar(deg, mse_list, width=0.4)
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")
        plt.title("MSE score of the models")
        plt.show()

    except Exception as e:
        print(e)
