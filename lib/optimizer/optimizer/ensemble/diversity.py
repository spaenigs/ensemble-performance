from itertools import combinations

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd


def kappa(y_pred_1, y_pred_2):
    a, b, c, d = 0, 0, 0, 0
    for i, j in zip(y_pred_1, y_pred_2):
        if (i, j) == (1, 1):
            a += 1
        elif (i, j) == (0, 0):
            d += 1
        elif (i, j) == (1, 0):
            b += 1
        elif (i, j) == (0, 1):
            c += 1
    a, b, c, d = \
        [v/len(y_pred_1) for v in [a, b, c, d]]
    dividend = 2 * (a*d-b*c)
    divisor = ((a+b) * (b+d)) + ((a+c) * (c+d))
    return dividend / divisor


def kappa_error_plot(meta_clf, X_list, y):
    train_index, test_index = \
        list(StratifiedKFold(n_splits=2).split(X_list[0], y))[0]

    X_train_list, X_test_list = \
        [X[train_index] for X in X_list], \
        [X[test_index] for X in X_list]
    y_train, y_test = y[train_index], y[test_index]

    meta_clf.fit(X_train_list, y_train)

    res = []
    for ((enc_name_1, clf_1), X_test_1), ((enc_name_2, clf_2), X_test_2) in \
            combinations(zip(meta_clf.estimators_, X_test_list), 2):
        y_pred_tree_1, y_pred_tree_2 = \
            clf_1.predict(X_test_1), clf_2.predict(X_test_2)
        error_1, error_2 = \
            1 - accuracy_score(y_pred_tree_1, y_test), \
            1 - accuracy_score(y_pred_tree_2, y_test)
        mean_pairwise_error = np.mean([error_1, error_2])
        k = kappa(y_pred_tree_1, y_pred_tree_2)
        res += [[k, mean_pairwise_error, f"{enc_name_1}_{enc_name_2}"]]

    return pd.DataFrame(res, columns=["kappa", "mean_pairwise_error", "encoding_pair"])
