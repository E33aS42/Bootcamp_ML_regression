import numpy as np


def check_input(y, y_hat, pos_label):
    try:
        assert isinstance(
            y, np.ndarray) and (y.ndim == 2 or y.ndim == 1), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert isinstance(
            y_hat, np.ndarray) and (y_hat.ndim == 2 or y_hat.ndim == 1), "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
        assert pos_label in y, "3rd argument must be an element of the array arguments"
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)
        assert y.shape[0] == y_hat.shape[0], "arrays must be the same size"
        m = y.shape[0]

        return 1

    except Exception as e:
        print(e)
        return 0


def calc_tp_fp_tn_fn(y, y_hat, pos_label=1):
    """
    A true positive (TP) is a datapoint we predicted positively that we were correct about.
    A true negative (TN) is a datapoint we predicted negatively that we were correct about.
    A false positive (FP) is a datapoint we predicted positively that we were incorrect about.
    A false negative (FN) is a datapoint we predicted negatively that we were incorrect about.
    """
    if check_input(y, y_hat, pos_label):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for yi, yhi in zip(y, y_hat):
            if yi == pos_label and yhi == pos_label:
                tp += 1
            elif yi != pos_label and yhi != pos_label:
                tn += 1
            elif yi != pos_label and yhi == pos_label:
                fp += 1
            elif yi == pos_label and yhi != pos_label:
                fn += 1
        return tp, tn, fp, fn


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Accuracy: tells you the percentage of predictions that are accurate (i.e. the correct class was predicted). Accuracy doesn't give information about either error type.

    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    Returns:
    The accuracy score as a float.
    None on any error.
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

        p = 0
        for yi, yhi in zip(y, y_hat):
            if yi == yhi:
                p += 1
        return p / len(y)

    except Exception as e:
        print(e)
        return 0


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Precision: tells you how much you can trust your model when it says that an object belongs to Class A. More precisely, it is the percentage of the objects assigned to Class A that really were A objects. You use precision when you want to control for False positives.

    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if check_input(y, y_hat, pos_label):
        tp, tn, fp, fn = calc_tp_fp_tn_fn(y, y_hat, pos_label)
        return tp / (tp + fp)


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Recall: tells you how much you can trust that your model is able to recognize ALL Class A objects. It is the percentage of all A objects that were properly classified by the model as Class A. You use recall when you want to control for False negatives.

    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The recall score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if check_input(y, y_hat, pos_label):
        tp, tn, fp, fn = calc_tp_fp_tn_fn(y, y_hat, pos_label)
        return tp / (tp + fn)


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    F1 score: combines precision and recall in one single measure. You use the F1 score when you want to control both False positives and False negatives.

    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if check_input(y, y_hat, pos_label):
        r = recall_score_(y, y_hat, pos_label)
        p = precision_score_(y, y_hat, pos_label)
        return 2 * p * r / (p + r)
