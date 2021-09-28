# Adapted from https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py

from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import _get_n_samples_bootstrap, _parallel_build_trees, ForestClassifier, \
    _accumulate_prediction
from sklearn.exceptions import DataConversionWarning
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils.fixes import _joblib_parallel_args, delayed
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.tree import DecisionTreeClassifier
from warnings import warn
from joblib import Parallel
from scipy.sparse import issparse

import numpy as np

import threading

MAX_INT = np.iinfo(np.int32).max


class RandomForestClassifier(ForestClassifier):

    def __init__(self,
                 n_estimators=100,
                 encoding_names=None,
                 *,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None):
        super().__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state", "ccp_alpha"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha

        if encoding_names is not None:
            if n_estimators % len(encoding_names) == 0:
                self.encoding_names = encoding_names * int((n_estimators / len(encoding_names)))
            else:
                raise ValueError(f"encoding_names % n_estimators != 0.")

        # self.estimators = list(zip(self.encoding_names, [self.base_estimator] * n_estimators))

    def fit(self, X_list, y, sample_weight=None):
        # Validate or convert input data
        if issparse(y):
            raise ValueError(
                "sparse multilabel-indicator for y is not supported."
            )

        X_list = [self._validate_data(X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE)[0]
                  for X in X_list]

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X_list[0])

        for X in X_list:
            if issparse(X):
                # Pre-sort indices to avoid that each individual tree of the
                # ensemble sorts the indices.
                X.sort_indices()

        # Remap output
        self.n_features_ = X_list[0].shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X_list[0].shape[0],
            max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state)
                     for _ in range(n_more_estimators)]

            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_build_trees)(
                    t, self, X_, y, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap)
                for (i, t), X_ in zip(enumerate(trees), X_list))

            self.estimators_.extend(zip(self.encoding_names, trees))

        if self.oob_score:
            self._set_oob_score(X_list[0], y)

            # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    # def encoding_importance_plot(self, X_list, encoding_forest=None):
    #     return super().encoding_importance_plot(encoding_forest=self, X_list=X_list)
    #
    # def kappa_error_plot(self, X_list, y_list, random_state=42, encoding_forest=None):
    #     return super().kappa_error_plot(X_list, y_list, random_state, encoding_forest=self)

    def predict_proba(self, X_list):
        check_is_fitted(self)
        # Check data
        X_list = [self._validate_X_predict_estimator(e[1], X) for e, X in zip(self.estimators_, X_list)]

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [np.zeros((X_list[0].shape[0], j), dtype=np.float64)
                     for j in np.atleast_1d(self.n_classes_)]

        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(e[1].predict_proba, X_test, all_proba,
                                            lock)
            for e, X_test in zip(self.estimators_, X_list))

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba

    def predict(self, X_list):
        proba = self.predict_proba(X_list)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            # all dtypes should be the same, so just take the first
            class_type = self.classes_[0].dtype
            predictions = np.empty((n_samples, self.n_outputs_),
                                   dtype=class_type)

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(
                    np.argmax(proba[k], axis=1), axis=0)

            return predictions

    def _validate_X_predict_estimator(self, estimator, X):
        """
        Validate X whenever one tries to predict, apply, predict_proba.
        Overrides _validate_X_predict.
        """
        check_is_fitted(self)
        return estimator._validate_X_predict(X, check_input=True)

    @property
    def feature_importances_(self):
        raise NotImplementedError