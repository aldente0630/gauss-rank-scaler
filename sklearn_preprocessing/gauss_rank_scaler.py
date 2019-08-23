import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.special import erf, erfinv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted


class GuassRankScaler(BaseEstimator, TransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance
        The standard score of a sample `x` is calculated as:
            z = (x - u) / s
        where `u` is the mean of the training samples or zero if `with_mean=False`,
        and `s` is the standard deviation of the training samples or one if
        `with_std=False`.
        Centering and scaling happen independently on each feature by computing
        the relevant statistics on the samples in the training set. Mean and
        standard deviation are then stored to be used on later data using the
        `transform` method.
        Standardization of a dataset is a common requirement for many
        machine learning estimators: they might behave badly if the
        individual features do not more or less look like standard normally
        distributed data (e.g. Gaussian with 0 mean and unit variance).
        For instance many elements used in the objective function of
        a learning algorithm (such as the RBF kernel of Support Vector
        Machines or the L1 and L2 regularizers of linear models) assume that
        all features are centered around 0 and have variance in the same
        order. If a feature has a variance that is orders of magnitude larger
        that others, it might dominate the objective function and make the
        estimator unable to learn from other features correctly as expected.
        This scaler can also be applied to sparse CSR or CSC matrices by passing
        `with_mean=False` to avoid breaking the sparsity structure of the data.
        Read more in the :ref:`User Guide <preprocessing_scaler>`.
        Parameters
        ----------
        copy : boolean, optional, default True
            If False, try to avoid a copy and do inplace scaling instead.
            This is not guaranteed to always work inplace; e.g. if the data is
            not a NumPy array or scipy.sparse CSR matrix, a copy may still be
            returned.
        with_mean : boolean, True by default
            If True, center the data before scaling.
            This does not work (and will raise an exception) when attempted on
            sparse matrices, because centering them entails building a dense
            matrix which in common use cases is likely to be too large to fit in
            memory.
        with_std : boolean, True by default
            If True, scale the data to unit variance (or equivalently,
            unit standard deviation).
        Attributes
        ----------
        scale_ : ndarray or None, shape (n_features,)
            Per feature relative scaling of the data. This is calculated using
            `np.sqrt(var_)`. Equal to ``None`` when ``with_std=False``.
            .. versionadded:: 0.17
               *scale_*
        mean_ : ndarray or None, shape (n_features,)
            The mean value for each feature in the training set.
            Equal to ``None`` when ``with_mean=False``.
        var_ : ndarray or None, shape (n_features,)
            The variance for each feature in the training set. Used to compute
            `scale_`. Equal to ``None`` when ``with_std=False``.
        n_samples_seen_ : int or array, shape (n_features,)
            The number of samples processed by the estimator for each feature.
            If there are not missing samples, the ``n_samples_seen`` will be an
            integer, otherwise it will be an array.
            Will be reset on new calls to fit, but increments across
            ``partial_fit`` calls.
        """

    def __init__(self, epsilon=1e-4, copy=True, n_jobs=None, interp_kind='linear', interp_copy=False):
        self.epsilon = epsilon
        self.copy = copy
        self.interp_params = {'kind': interp_kind, 'copy': interp_copy, 'fill_value': 'extrapolate'}
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        """
        X = check_array(X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)

        self.interp_func_ = Parallel(n_jobs=self.n_jobs)(delayed(self._fit)(x) for x in X.T)
        return self

    def _fit(self, x):
        x = self.drop_duplicates(x)
        rank = np.argsort(np.argsort(x))
        bound = 1.0 - self.epsilon
        factor = np.max(rank) / 2.0 * bound
        scaled_rank = np.clip(rank / factor - bound, -bound, bound)
        return interp1d(x, scaled_rank, **self.interp_params)

    def transform(self, X, copy=None):
        """Perform standardization by centering and scaling
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        check_is_fitted(self, 'interp_func_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)

        X = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._transform)(i, x) for i, x in enumerate(X.T))).T
        return X

    def _transform(self, i, x):
        return erfinv(self.interp_func_[i](x))

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        check_is_fitted(self, 'interp_func_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)

        X = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._inverse_transform)(i, x) for i, x in enumerate(X.T))).T
        return X

    def _inverse_transform(self, i, x):
        inv_interp_func = interp1d(self.interp_func_[i].y, self.interp_func_[i].x, **self.interp_params)
        return inv_interp_func(erf(x))

    @staticmethod
    def drop_duplicates(x):
        k = np.zeros_like(x, dtype=bool)
        k[np.unique(x, return_index=True)[1]] = True
        return x[k]
