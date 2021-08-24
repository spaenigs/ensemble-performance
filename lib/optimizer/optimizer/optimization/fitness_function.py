from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold

import pandas as pd
import numpy as np


def train_ensemble(
        individual,
        paths_to_encoded_datasets,
        train_index,
        meta_clf,
        base_clf,
        random_state
):
    paths = np.array(paths_to_encoded_datasets)[np.nonzero(individual)[0]]
    if len(paths) > 11 or len(paths) <= 1:
        mcc = 0.0
    else:
        encoded_datasets = \
            [pd.read_csv(p, index_col=0).loc[train_index, :]
             for p in paths]
        X = encoded_datasets[0].iloc[:, :-1].values
        y = encoded_datasets[0]["y"].values
        mcc = 0
        rkf = RepeatedStratifiedKFold(n_repeats=1, n_splits=5)
        for train_index_i, test_index_i in rkf.split(X, y):
            X_train_list, X_test_list = \
                [df.iloc[train_index_i, :-1].values for df in encoded_datasets], \
                [df.iloc[test_index_i, :-1].values for df in encoded_datasets]
            y_train, y_test = y[train_index_i], y[test_index_i]
            meta_clf.estimators = [(paths[i], base_clf) for i in range(len(paths))]
            meta_clf.n_jobs = 1
            try:
                meta_clf.fit(X_train_list, y_train)
                y_pred = meta_clf.predict(X_test_list)
                mcc_ = matthews_corrcoef(y_test, y_pred)
            except np.linalg.LinAlgError as e:
                print(e)
                mcc_ = 0.0
            except ValueError as e:
                print(e)
                mcc_ = 0.0
            if mcc_ < 0:
                mcc += 0
            else:
                mcc += mcc_
    return 1 - (mcc/5), {"mcc": mcc/5}
