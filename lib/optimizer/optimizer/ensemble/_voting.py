from abc import abstractmethod
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble._voting import _BaseVoting
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import _deprecate_positional_args, Bunch
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from ._base import _fit_single_estimator

import numpy as np


class _MyBaseVoting(_BaseVoting):

    @abstractmethod
    def fit(self, X_list, y, sample_weight=None):
        """Get common fit operations."""
        names, clfs = self._validate_estimators()

        if (self.weights is not None and
                len(self.weights) != len(self.estimators)):
            raise ValueError('Number of `estimators` and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(
                clone(clf), X_list[idx], y,
                sample_weight=sample_weight,
                message_clsname='Voting',
                message=self._log_message(names[idx],
                                          idx + 1, len(clfs))
            )
            for idx, clf in enumerate(clfs) if clf != 'drop'
            # for idx, clf in enumerate(clfs) if clf != 'drop'
        )

        self.estimators_ = list(zip(names, self.estimators_))

        self.named_estimators_ = Bunch()

        # Uses 'drop' as placeholder for dropped estimators
        est_iter = iter(self.estimators_)
        for name, est in self.estimators:
            current_est = est if est == 'drop' else next(est_iter)
            self.named_estimators_[name] = current_est

        return self


class VotingClassifier(ClassifierMixin, _MyBaseVoting):

    @_deprecate_positional_args
    def __init__(self, estimators, *, voting='hard', weights=None,
                 n_jobs=None, flatten_transform=True, verbose=False):
        super().__init__(estimators=estimators)
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose

    def fit(self, X_list, y, sample_weight=None):
        check_classification_targets(y)
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        transformed_y = self.le_.transform(y)

        return super().fit(X_list, transformed_y, sample_weight)

    def _predict(self, X_list):
        """Collect results from clf.predict calls."""
        return np.asarray([est.predict(X_list[idx])
                           for idx, (_, est) in
                           enumerate(self.estimators_)]).T

    def predict(self, X_list):
        """Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        maj : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        if self.voting == 'soft':
            maj = np.argmax(self.predict_proba(X_list), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X_list)
            maj = np.apply_along_axis(
                lambda x: np.argmax(
                    np.bincount(x, weights=self._weights_not_none)),
                axis=1, arr=predictions)

        maj = self.le_.inverse_transform(maj)

        return maj

    def _collect_probas(self, X_list):
        """Collect results from clf.predict calls."""
        return np.asarray([est.predict_proba(X_list[idx])
                           for idx, (_, est) in
                           enumerate(self.estimators_)])
                           # zip(self.named_estimators.keys(), self.estimators_)])

    def _predict_proba(self, X_list):
        """Predict class probabilities for X in 'soft' voting."""
        check_is_fitted(self)
        avg = np.average(self._collect_probas(X_list), axis=0,
                         weights=self._weights_not_none)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        avg : array-like of shape (n_samples, n_classes)
            Weighted average probability for each class per sample.
        """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        return self._predict_proba