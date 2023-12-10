import numpy as np
import pandas as pd


def confusion_matrix_(y, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    labels: optional, a list of labels to index the matrix.
    This may be used to reorder or select a subset of labels. (default=None)
    df_option: optional, if set to True the function will return a pandas DataFrame
    instead of a numpy array. (default=False)
    Return:
    The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
    None if any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        assert isinstance(
            y, np.ndarray) and (y.ndim == 2 or y.ndim == 1), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(
            y_hat, np.ndarray) and (y_hat.ndim == 2 or y_hat.ndim == 1), "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)
        assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"
        m = y.shape[0]

        # create labels list
        list_labels = []
        for i in y:
            if i[0] not in list_labels:
                list_labels.append(i[0])
        for i in y_hat:
            if i[0] not in list_labels:
                list_labels.append(i[0])
        list_labels.sort()

        if labels is not None:
            labels.sort()
        else:
            labels = list_labels

        # a dictionary to store predicted and actual values
        dict_labels = {}
        for i in list_labels:
            dict_labels[i] = 0

        # The confusion matrix to build
        conf_mat = pd.DataFrame(0, columns=list_labels, index=list_labels)

        # now let's count all our values
        for yi, yhi in zip(y, y_hat):
            conf_mat.loc[yi, yhi] += 1
        conf_mat = conf_mat.loc[labels, labels]

        if df_option == False:
            return conf_mat.values
        return conf_mat

    except Exception as e:
        print(e)
        return 0
