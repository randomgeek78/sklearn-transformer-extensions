from functools import partial
from scipy import sparse
from sklearn.base import clone
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import _get_column_indices, _safe_indexing
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import (
    check_is_fitted,
    _check_feature_names_in,
    _make_indexable,
)
import numpy as np
from typing import Iterable, Any
from sklearn.utils import _determine_key_type

try:
    import pandas as pd
except:
    pass

import re
from sklearn.exceptions import NotFittedError


class XyData:

    def __init__(self, X, y):

        self.X, self.y = _make_indexable((X, y))

    def __getitem__(self, ind):

        X = _safe_indexing(self.X, ind)
        y = _safe_indexing(self.y, ind)

        return XyData(X, y)

    def __len__(self):

        return len(self.X)

    def __iter__(self):

        return iter((self.X, self.y))

    def __repr__(self):

        tx = 'numpy'
        if hasattr(self.X, "iloc"):
            tx = 'pandas'
        sx = self.X.shape

        ty = 'numpy'
        if hasattr(self.y, "iloc"):
            ty = 'pandas'
        sy = self.y.shape

        format = dict(name=self.__class__.__name__, tx=tx, sx=sx, ty=ty, sy=sy)
        return '{name}(X={tx}({sx}), y={ty}({sy}))'.format(**format)


class XyAdapterStub(object):

    def __call__(self, klass):
        obj = XyAdapterStub()
        obj.__class__ = XyAdapterFactory(klass)
        return obj


class XyAdapterBase:
    pass


def XyAdapterFactory(klass):

    # https://stackoverflow.com/questions/4647566/pickle-a-dynamically-parameterized-sub-class
    class XyAdapter(klass, XyAdapterBase):
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
        >>> from sklearn_transformer_extensions import XyData
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.linear_model import LogisticRegression
        >>> import numpy as np
        >>> X, y = load_iris(return_X_y=True)
        >>> Xy = XyData(X, y)
        >>> clf = XyAdapter(LogisticRegression(random_state=0))
        >>> clf.fit(Xy)
        XyAdapter(transformer=LogisticRegression(random_state=0))
        >>> clf.predict(X[:2, :])
        array([0., 0.])
        >>> clf.predict_proba(X[:2, :])
        array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
               [9.7...e-01, 2.8...e-02, ...e-08]])
        >>> clf.score(Xy)
        0.97...
        """

        def __init__(self, transformer, *, ofmt=None):

            self.transformer = transformer

            assert ofmt in ['pandas', 'numpy', None]
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
                return object.__getattribute__(
                    object.__getattribute__(self, "transformer"), name)

        def __hash__(self):
            return hash((self.transformer, self.ofmt))

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

        def _joinXy(self, X, y, ofmt):

            if ofmt == 'pandas':

                columns = None
                # Check if column names can be obtained from instance methods
                for attr in ["get_feature_names_out", "get_feature_names"]:
                    try:
                        columns = getattr(self.transformer, attr)()
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

                if y is not None:
                    X = XyData(X, y)

            else:

                if y is not None:
                    X = XyData(X, y)

            return X

        def _call(self, method, X, requires_y=True, join_y=True, reset=True,
                  **params):

            y = None
            if type(X) == XyData:
                X, y = X.X, X.y

            if hasattr(self, "_check_feature_names"):
                self._check_feature_names(X, reset=reset)
            if hasattr(self, "_check_n_features"):
                self._check_n_features(X, reset=reset)

            ofmt = self._check_ofmt(X)

            # if reset:
            #     self.transformer_ = clone(self.transformer)
            method_fn = getattr(self.transformer, method)
            if requires_y:
                method_fn = partial(method_fn, X, y, **params)
            else:
                method_fn = partial(method_fn, X, **params)

            if join_y:
                X = method_fn()
                return self._joinXy(X, y, ofmt)
            else:
                ret = method_fn()
                if ret is self.transformer:
                    return self
                else:
                    return ret

        def __sklearn_is_fitted__(self):
            """Indicate whether pipeline has been fit."""
            try:
                # check if the last step of the pipeline is fitted
                # we only check the last step since if the last step is fit, it
                # means the previous steps should also be fit. This is faster than
                # checking if every step of the pipeline is fit.
                check_is_fitted(self.transformer)
                return True
            except NotFittedError:
                return False

        @if_delegate_has_method("transformer")
        def fit_transform(self, X, _=None, **fit_params):

            return self._call("fit_transform", X, requires_y=True, join_y=True,
                              reset=True, **fit_params)

        @if_delegate_has_method("transformer")
        def fit(self, X, _=None, **fit_params):

            return self._call("fit", X, requires_y=True, join_y=False,
                              reset=True, **fit_params)

        @if_delegate_has_method("transformer")
        def transform(self, X):

            return self._call("transform", X, requires_y=False, join_y=True,
                              reset=False)

        @if_delegate_has_method("transformer")
        def predict(self, X, **predict_params):

            return self._call("predict", X, requires_y=False, join_y=False,
                              reset=False, **predict_params)

        @if_delegate_has_method("transformer")
        def predict_proba(self, X, **predict_proba_params):

            return self._call("predict_proba", X, requires_y=False,
                              join_y=False, reset=False, **predict_proba_params)

        @if_delegate_has_method("transformer")
        def predict_log_proba(self, X, **predict_proba_params):

            return self._call("predict_log_proba", X, requires_y=False,
                              join_y=False, reset=False, **predict_proba_params)

        @if_delegate_has_method("transformer")
        def score(self, X, _=None, **score_params):

            return self._call("score", X, requires_y=True, join_y=False,
                              reset=False, **score_params)

        @if_delegate_has_method("transformer")
        def score_samples(self, X):

            return self._call("score_samples", X, requires_y=False,
                              join_y=False, reset=False)

        @if_delegate_has_method("transformer")
        def decision_function(self, X):

            return self._call("decision_function", X, requires_y=False,
                              join_y=False, reset=False)

        def __reduce__(self):
            return (XyAdapterStub(), (klass, ), self.__dict__)

    return XyAdapter


def XyAdapter(transformer, *, ofmt=None):
    klass = transformer.__class__
    return XyAdapterFactory(klass)(transformer, ofmt=ofmt)
