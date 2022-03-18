from functools import partial
from scipy import sparse
from sklearn.base import clone
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import _get_column_indices, _safe_indexing
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import (
    check_is_fitted,
    _check_feature_names_in,
    _num_features,
)
import numpy as np
from typing import Iterable, Any
from sklearn.utils import _determine_key_type

try:
    import pandas as pd
except:
    pass

import re


class XyAdapter(TransformerMixin, BaseEstimator):
    """An adapter class that enables interaction with scikit-learn transformers
    and estimators using an input numpy array or pandas dataframe that contains
    both features (X) and labels (y).

    Internally, for any method call, the adapter splits the input into features
    (X) and labels (y) and transparently calls the same method in the underlying
    transformer or estimator instance (if exists) with the split features and
    labels (as needed).

    The returned object from the call to the underlying transformer or
    estimator is forwarded to the external caller. For `fit_transform` and
    `transform` method calls, the returned object is the transformed features
    (X). For these two methods, the transformed features are combined with the
    labels (y) before returning to the external caller.

    For the `fit_transform` and `transform` method calls, the user can specify
    if the output should be a numpy array or a pandas dataframe. If the output
    format is pandas dataframe, then the underlying transformer's
    get_feature_names_out or get_feature_names method is called (if exists) to
    infer the column names of the returned object. If these don't exist, then
    the column names from the input dataframe are used. We run into an error if
    all these following conditions hold: a) the requested format is a
    dataframe, b) transformer does not provide either the get_feature_names_out
    or the get_feature_names methods, c) the transformed object contains a
    different number of columns from the input dataframe/array. 

    Parameters
    ----------
    transformer: a single estimator or transformer instance, required
        The estimator or group of estimators to be cloned.
    target_col : list[str], list[int], str, int or None, default=None
        The columns to pry away from the input X that correspond to the labels
        (y) before calling the underlying transformer.
    ofmt: 'pandas', 'numpy', None, default=None
        The output format for returns from methods. 'pandas' returns either a
        DataFrame or Series. 'numpy' returns a 2-d or 1-d numpy array. None
        keeps the output format the same as the input. 
        
    Attributes
    ----------
    transformer_: a fitted instance of the estimator or transformer

    Examples
    --------

    In this example, we recreate the example from scikit-learn's
    LogisticRegression documentation. We directly work with the train
    datastructure that contains both X and y.

    >>> from sklearn_transformer_extensions import XyAdapter
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X, y = load_iris(return_X_y=True)
    >>> train = np.hstack((X, y.reshape((-1, 1))))
    >>> clf = XyAdapter(LogisticRegression(random_state=0), target_col=4)
    >>> clf.fit(train)
    XyAdapter(target_col=[4], transformer=LogisticRegression(random_state=0))
    >>> clf.predict(X[:2, :])
    array([0., 0.])
    >>> clf.predict_proba(X[:2, :])
    array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
           [9.7...e-01, 2.8...e-02, ...e-08]])
    >>> clf.score(train)
    0.97...
    """

    def __init__(self, transformer, target_col=None, *, ofmt=None):

        self.transformer = transformer
        # repr(self.transformer)

        assert isinstance(target_col, (type(None), str, int, slice, Iterable))
        self.target_col = target_col

        assert isinstance(ofmt, (type(None), str))
        if isinstance(ofmt, str):
            assert ofmt in ['pandas', 'numpy']
        self.ofmt = ofmt

    @if_delegate_has_method("transformer")
    def get_params(self, deep=True):
        return self.transformer.get_params(deep=deep)

    @if_delegate_has_method("transformer")
    def set_params(self, **kwargs):
        self.transformer.set_params(**kwargs)
        return self

    def __repr__(self, N_CHAR_MAX=700):
        if isinstance(self.transformer, BaseEstimator):
            return self.transformer.__repr__(N_CHAR_MAX)
        return self.transformer.__repr__()

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            try:
                return object.__getattribute__(
                    object.__getattribute__(self, "transformer_"), name)
            except AttributeError:
                return object.__getattribute__(
                    object.__getattribute__(self, "transformer"), name)

    def __hash__(self):
        return hash((self.transformer, self.target_col, self.ofmt))

    def __eq__(self, other):
        if isinstance(other, self.transformer.__class__):
            return self.transformer == other
        else:
            return hash(self) == hash(other)

    def _check_ofmt(self, X):

        ofmt = self.ofmt
        if self.ofmt is None:
            ofmt = 'pandas' if hasattr(X, "iloc") else 'numpy'

        return ofmt

    def _splitXy(self, X, reset=True):
        """Split Xy columns into X and y.

        Parameters
        ----------
        X: 2-d array or DataFrame

        Returns
        -------
        Returns X and y after splitting
        """

        y = None
        if self.target_col is not None:

            if reset:
                self.n_features_xy_ = _num_features(X)

            if self.n_features_xy_ == X.shape[1]:
                features = np.full((X.shape[1], ), False)

                target_col = self.target_col
                if isinstance(self.target_col, (str, int)):
                    target_col = [self.target_col]
                elif isinstance(self.target_col, Iterable):
                    target_col = np.r_[self.target_col]
                target_col = target_col if target_col is not None else []
                y_cols = _get_column_indices(X, target_col)
                features[y_cols] = True

                y = _safe_indexing(X, features, axis=1)
                X = _safe_indexing(X, ~features, axis=1)

                if y.size == 0:
                    y = None

                if sparse.issparse(y):
                    y = y.toarray()

                if y is not None:
                    y = y.squeeze()  # type: ignore

                dtype = None
                if hasattr(X, "iloc"):
                    dtype_id = np.unique([x.kind for x in X.dtypes])
                    if len(dtype_id) == 1:
                        dtype = dtype_id[0]
                else:
                    dtype = X.dtype.kind

                dtype_map = {
                    "i": "int",
                    "u": "int",
                    "b": "bool",
                    "O": "str",
                    "U": "str",
                    "S": "str",
                }

                if dtype in dtype_map and dtype_map[dtype] == 'str':
                    X = X.squeeze()

            else:
                y = None

        self._check_n_features(X, reset=reset)
        self._check_feature_names(X, reset=reset)

        return X, y

    def _joinXy(self, X, y, ofmt):

        # if sparse.issparse(X):
        #     X = X.todense()

        if ofmt == 'pandas':

            columns = None
            # Check if column names can be obtained from instance methods
            for attr in ["get_feature_names_out", "get_feature_names"]:
                try:
                    columns = getattr(self.transformer_, attr)()
                    break
                except:
                    pass
            # Else, default to what can be inferred from the input
            if columns is None:
                columns = _check_feature_names_in(self)

            if not hasattr(X, "iloc"):
                X = pd.DataFrame(X, columns=columns).infer_objects()
            else:
                X.columns = columns  # type: ignore

            if y is None:
                y = pd.DataFrame(index=X.index)
            elif not hasattr(y, "iloc"):
                assert self.target_col is not None
                if y.ndim == 1:
                    y = np.atleast_2d(y).T
                columns = ["y" + str(i) for i in range(y.shape[1])]
                y = pd.DataFrame(y, columns=columns)

            Xy = pd.concat((X, y), axis=1)

        else:

            Xy = X
            if y is not None:
                assert (not sparse.issparse(y)) or (y.ndim == 2)
                if y.ndim <= 1:
                    y = np.atleast_2d(y).T
                if any([sparse.issparse(x) for x in (X, y)]):
                    Xy = sparse.hstack((X, y), format='csr')
                else:
                    Xy = np.hstack((X, y))  # type: ignore

        return Xy

    def _call(self, method, X, requires_y=True, join_y=True, reset=True,
              **params):

        X, y = self._splitXy(X, reset=reset)
        ofmt = self._check_ofmt(X)

        if reset:
            self.transformer_ = clone(self.transformer)
        method_fn = getattr(self.transformer_, method)
        if requires_y:
            method_fn = partial(method_fn, X, y, **params)
        else:
            method_fn = partial(method_fn, X, **params)

        if join_y:
            X = method_fn()
            return self._joinXy(X, y, ofmt)
        else:
            ret = method_fn()
            if type(ret) == type(self.transformer_) \
                    and ret == self.transformer_:
                return self
            else:
                return ret

    @if_delegate_has_method("transformer")
    def fit_transform(self, X, _=None, **fit_params):

        return self._call("fit_transform", X, requires_y=True, join_y=True,
                          reset=True, **fit_params)

    @if_delegate_has_method("transformer")
    def fit(self, X, _=None, **fit_params):

        return self._call("fit", X, requires_y=True, join_y=False, reset=True,
                          **fit_params)

    @if_delegate_has_method("transformer")
    def transform(self, X):

        check_is_fitted(self)
        return self._call("transform", X, requires_y=False, join_y=True,
                          reset=False)

    @if_delegate_has_method("transformer")
    def predict(self, X, **predict_params):

        check_is_fitted(self)
        return self._call("predict", X, requires_y=False, join_y=False,
                          reset=False, **predict_params)

    @if_delegate_has_method("transformer")
    def predict_proba(self, X, **predict_proba_params):

        check_is_fitted(self)
        return self._call("predict_proba", X, requires_y=False, join_y=False,
                          reset=False, **predict_proba_params)

    @if_delegate_has_method("transformer")
    def predict_log_proba(self, X, **predict_proba_params):

        check_is_fitted(self)
        return self._call("predict_log_proba", X, requires_y=False,
                          join_y=False, reset=False, **predict_proba_params)

    @if_delegate_has_method("transformer")
    def score(self, X, _=None, **score_params):

        check_is_fitted(self)
        return self._call("score", X, requires_y=True, join_y=False,
                          reset=False, **score_params)

    @if_delegate_has_method("transformer")
    def decision_function(self, X):

        check_is_fitted(self)
        return self._call("decision_function", X, requires_y=False,
                          join_y=False, reset=False)
