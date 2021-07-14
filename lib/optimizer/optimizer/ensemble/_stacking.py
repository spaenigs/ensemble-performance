from copy import deepcopy
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin, is_classifier, clone
from sklearn.ensemble._stacking import _BaseStacking
from sklearn.model_selection import check_cv, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import _deprecate_positional_args, Bunch
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
from ._base import _fit_single_estimator

import numpy as np


class _MyBaseStacking(_BaseStacking):
    """Base class for stacking method."""

    def __init__(self, estimators, final_estimator=None, *, cv=None,
                 stack_method='auto', n_jobs=None, verbose=0,
                 passthrough=False):
        super().__init__(estimators=estimators)
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.passthrough = passthrough

    def _validate_final_estimator(self):
        self._clone_final_estimator(default=LogisticRegression())
        if not is_classifier(self.final_estimator_):
            raise ValueError(
                "'final_estimator' parameter should be a classifier. Got {}"
                .format(self.final_estimator_)
            )

    def fit(self, X_list, y, sample_weight=None):
        # all_estimators contains all estimators, the one to be fitted and the
        # 'drop' string.
        names, all_estimators = self._validate_estimators()
        self._validate_final_estimator()

        stack_method = [self.stack_method] * len(all_estimators)

        # Fit the base estimators on the whole training data. Those
        # base estimators will be used in transform, predict, and
        # predict_proba. They are exposed publicly.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(clone(est), X_list[idx], y, sample_weight)
            for idx, est in enumerate(all_estimators) if est != 'drop'
            # for est in all_estimators if est != 'drop'
        )

        self.estimators_ = list(zip(names, self.estimators_))

        self.named_estimators_ = Bunch()
        est_fitted_idx = 0
        for name_est, org_est in zip(names, all_estimators):
            if org_est != 'drop':
                self.named_estimators_[name_est] = self.estimators_[est_fitted_idx]
                est_fitted_idx += 1
            else:
                self.named_estimators_[name_est] = 'drop'

        # To train the meta-classifier using the most data as possible, we use
        # a cross-validation to obtain the output of the stacked estimators.

        # To ensure that the data provided to each estimator are the same, we
        # need to set the random state of the cv if there is one and we need to
        # take a copy.
        cv = check_cv(self.cv, y=y, classifier=is_classifier(self))
        if hasattr(cv, 'random_state') and cv.random_state is None:
            cv.random_state = np.random.RandomState()

        self.stack_method_ = [
            self._method_name(name, est, meth)
            for name, est, meth in zip(names, all_estimators, stack_method)
        ]
        fit_params = ({"sample_weight": sample_weight}
                      if sample_weight is not None
                      else None)
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(cross_val_predict)(clone(est), X_list[idx], y, cv=deepcopy(cv),
                                       method=meth, n_jobs=self.n_jobs,
                                       fit_params=fit_params,
                                       verbose=self.verbose)
            for idx, (est, meth) in enumerate(zip(all_estimators, self.stack_method_)) if est != 'drop'
            # for est, meth in zip(all_estimators, self.stack_method_)
            # if est != 'drop'
        )

        # Only not None or not 'drop' estimators will be used in transform.
        # Remove the None from the method as well.
        self.stack_method_ = [
            meth for (meth, est) in zip(self.stack_method_, all_estimators)
            if est != 'drop'
        ]

        X_meta = self._concatenate_predictions(X_list, predictions)
        _fit_single_estimator(self.final_estimator_, X_meta, y,
                              sample_weight=sample_weight)

        return self


class StackingClassifier(ClassifierMixin, _MyBaseStacking):

    @_deprecate_positional_args
    def __init__(self, estimators, final_estimator=None, *, cv=None,
                 stack_method='auto', n_jobs=None, passthrough=False,
                 verbose=0):
        super().__init__(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method=stack_method,
            n_jobs=n_jobs,
            passthrough=passthrough,
            verbose=verbose
        )

    def _validate_final_estimator(self):
        self._clone_final_estimator(default=LogisticRegression())
        if not is_classifier(self.final_estimator_):
            raise ValueError(
                "'final_estimator' parameter should be a classifier. Got {}"
                    .format(self.final_estimator_)
            )

    def fit(self, X_list, y, sample_weight=None):
        """Fit the estimators.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.
        Returns
        -------
        self : object
        """
        check_classification_targets(y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        return super().fit(X_list, self._le.transform(y), sample_weight)

    @if_delegate_has_method(delegate='final_estimator_')
    def predict(self, X_list, **predict_params):
        """Predict target for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            accounts for uncertainty in the final estimator.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """
        y_pred = super().predict(X_list, **predict_params)
        print("My prediction...")
        return self._le.inverse_transform(y_pred)

    @if_delegate_has_method(delegate='final_estimator_')
    def predict(self, X, **predict_params):
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            accounts for uncertainty in the final estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """

        check_is_fitted(self)
        return self.final_estimator_.predict(
            self.transform(X), **predict_params
        )

    def transform(self, X_list):
        """Return class labels or probabilities for X for each estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators) or \
                (n_samples, n_classes * n_estimators)
            Prediction outputs for each estimator.
        """
        return self._transform(X_list)

    def _transform(self, X_list):
        """Concatenate and return the predictions of the estimators."""
        check_is_fitted(self)
        predictions = [
            getattr(est[1], meth)(X_list[idx])
            for idx, (est, meth) in enumerate(zip(self.estimators_, self.stack_method_))
            if est != 'drop'
        ]
        return self._concatenate_predictions(X_list, predictions)
