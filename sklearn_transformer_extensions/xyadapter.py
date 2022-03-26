import sys as _sys
import warnings
from functools import partial
from scipy import sparse
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import (
    _make_indexable,
    _num_features,
)
import numpy as np
from typing import Dict, Any
from . import __version__

try:
    import pandas as pd
except:
    pass
import copy

from .xydata import XyData


class XyAdapterStub(object):

    def __call__(self, klass):
        obj = XyAdapterStub()
        obj.__class__ = XyAdapterFactory(klass)
        return obj


class XyAdapterBase:

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()

        if type(self).__module__.startswith("sklearn_transformer_extensions."):
            return dict(state.items(), _xyadapter_version=__version__)
        else:
            return state

    def __setstate__(self, state):
        if type(self).__module__.startswith("sklearn_transformer_extensions."):
            pickle_version = state.pop("_xyadapter_version", "pre-0.18")
            if pickle_version != __version__:
                warnings.warn(
                    "Trying to unpickle estimator {0} from version {1} when "
                    "using version {2}. This might lead to breaking code or "
                    "invalid results. Use at your own risk. "
                    "For more info please refer to:\n"
                    "https://scikit-learn.org/stable/modules/model_persistence"
                    ".html#security-maintainability-limitations".format(
                        self.__class__.__name__, pickle_version, __version__),
                    UserWarning,
                )

        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)


def _check_method(method):

    def fn(self):
        for klass in self.__class__.mro():
            if issubclass(klass, XyAdapterBase):
                continue
            break
        return hasattr(klass, method)  # type: ignore

    return fn


def XyAdapterFactory(klass):
    """An adapter that specializes a given klass object (which expected to
    be a scikit-learn transformer or estimator class) so all of klass'
    methods like `fit`, `transform`, etc now accept a XyData object in
    addition to accepting X and y as separate arguments (default behavior).

    Internally, if the input to a method is an XyData object, the adapter
    splits the input into features (X) and labels (y) before calling the
    corresponding scikit-learn object's method. If the input is not an
    XyData object, then the X and y arguments to the function are passed
    through as is effecting scikit-learn's traditional behavior.

    For transformers, the returned value from scikit-learn object's
    `fit_transform` and `transform` method calls are combined with labels
    (if exists) to create new XyData object and returned. If the original
    features (X) was pandas `DataFrame`, the returned transformed features
    is also a pandas `DataFrame`. The column names are obtained from
    scikit-learn's new `get_feature_names_out` interface. If scikit-learn's
    object does not provide this method, then we retain the original
    DataFrame's columns.

    Parameters
    ----------
    Same as the base class which is expected to be a scikit-learn
    transformer or estimator.
        
    Attributes
    ----------
    Same as the base class.

    Examples
    --------

    In this example, we recreate the example from scikit-learn's
    LogisticRegression documentation.

    >>> from sklearn_transformer_extensions import XyAdapter, XyData
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X, y = load_iris(return_X_y=True)
    >>> Xy = XyData(X, y)
    >>> clf = XyAdapter(LogisticRegression)(random_state=0)
    >>> clf.fit(Xy)
    LogisticRegression(random_state=0)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :])
    array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
           [9.7...e-01, 2.8...e-02, ...e-08]])
    >>> clf.score(Xy)
    0.97...

    """

    # https://stackoverflow.com/questions/4647566/pickle-a-dynamically-parameterized-sub-class
    class XyAdapter(klass, XyAdapterBase):

        @available_if(_check_method("get_params"))
        def get_params(self, deep: bool = True) -> Dict[str, Any]:
            # from xgboost/get_params
            params = super().get_params(deep)
            cp = copy.copy(self)
            cp.__class__ = cp.__class__.__bases__[0]
            params.update(cp.__class__.get_params(cp, deep))
            # if kwargs is a dict, update params accordingly
            if hasattr(self, "kwargs") and isinstance(self.kwargs, dict):
                params.update(self.kwargs)
            return params

        def __eq__(self, other):
            return (isinstance(self, other.__class__) or isinstance(
                other, self.__class__)) and self.__dict__ is other.__dict__

        def _joinXy(self, X, y, ofmt):

            if type(X) == XyData:
                # the input y replaced by what is got from the transformer
                X, y = X 

            if ofmt == 'pandas':

                if not hasattr(X, "iloc"):
                    feature_names_out = self._get_feature_names_out()
                    if feature_names_out is None:
                        Xt = X
                        if hasattr(X, "to_frame"):
                            Xt = X.to_frame()
                        elif sparse.issparse(X):
                            # Sparse already is 2-d
                            pass
                        else:
                            Xt = np.atleast_2d(X)
                        n_features = _num_features(Xt)
                        feature_names_out = np.asarray(
                            [f"col{i}" for i in range(n_features)],
                            dtype=object)

                    if sparse.issparse(X):
                        X = pd.DataFrame.sparse.from_spmatrix(
                            X, columns=feature_names_out)
                        X = X.infer_objects()
                    else:
                        X = pd.DataFrame(X, columns=feature_names_out)
                        X = X.infer_objects()

                if y is not None and hasattr(y, "iloc") is False:
                    yt = y
                    if hasattr(y, "to_frame"):
                        yt = y.to_frame()
                    elif sparse.issparse(y):
                        # Sparse already is 2-d
                        pass
                    else:
                        yt = np.atleast_2d(y)
                    n_features = _num_features(yt)
                    feature_names_out = np.asarray(
                        [f"y{i}" for i in range(n_features)], dtype=object)

                    if sparse.issparse(y):
                        y = pd.DataFrame.sparse.from_spmatrix(
                            y, columns=feature_names_out)
                        y = y.infer_objects()
                    else:
                        y = pd.DataFrame(y, columns=feature_names_out)
                        y = y.infer_objects()
                    y = y.squeeze()

                    # Sync the indices of X and y. y takes on the indices of x
                    # assuming they are in sync
                    y = y.reset_index(drop=True).reindex_like(X)

            return XyData(X, y)

        def _call(self, method, X, y=None, requires_y=True, join_y=True,
                  reset=True, **params):

            input_type = type(X)

            if input_type == XyData:
                X, y = X

            try:
                klass = self.__class__
                self.__class__ = klass.__bases__[0]
                method_fn = partial(getattr(self.__class__, method), self)
                if requires_y:
                    method_fn = partial(method_fn, X, y, **params)
                else:
                    method_fn = partial(method_fn, X, **params)
                ret = method_fn()
            finally:
                self.__class__ = klass  # type: ignore

            if join_y and input_type == XyData:
                ofmt = 'pandas' if hasattr(X, "iloc") else 'numpy'
                ret = self._joinXy(ret, y, ofmt)

            return ret

        def _get_feature_names_out(self, input_features=None):

            try:
                return self.get_feature_names_out(input_features)
            except AttributeError:
                pass
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore',
                                          category=FutureWarning)
                    return self.get_feature_names()
            except AttributeError:
                pass
            return None

        @available_if(_check_method("fit_transform"))
        def fit_transform(self, X, y=None, **fit_params):

            return self._call("fit_transform", X, y, requires_y=True,
                              join_y=True, reset=True, **fit_params)

        @available_if(_check_method("fit"))
        def fit(self, X, y=None, **fit_params):

            return self._call("fit", X, y, requires_y=True, join_y=False,
                              reset=True, **fit_params)

        @available_if(_check_method("transform"))
        def transform(self, X, *args, **kwargs):

            return self._call("transform", X, requires_y=False, join_y=True,
                              reset=False)

        @available_if(_check_method("predict"))
        def predict(self, X, **predict_params):

            return self._call("predict", X, requires_y=False, join_y=False,
                              reset=False, **predict_params)

        @available_if(_check_method("predict_proba"))
        def predict_proba(self, X, **predict_proba_params):

            return self._call("predict_proba", X, requires_y=False,
                              join_y=False, reset=False, **predict_proba_params)

        @available_if(_check_method("predict_log_proba"))
        def predict_log_proba(self, X, **predict_proba_params):

            return self._call("predict_log_proba", X, requires_y=False,
                              join_y=False, reset=False, **predict_proba_params)

        @available_if(_check_method("score"))
        def score(self, X, y=None, **score_params):

            return self._call("score", X, y, requires_y=True, join_y=False,
                              reset=False, **score_params)

        @available_if(_check_method("score_samples"))
        def score_samples(self, X):

            return self._call("score_samples", X, requires_y=False,
                              join_y=False, reset=False)

        @available_if(_check_method("decision_function"))
        def decision_function(self, X):

            return self._call("decision_function", X, requires_y=False,
                              join_y=False, reset=False)

        def __reduce__(self):
            return (XyAdapterStub(), (klass, ), self.__getstate__())

    # https://hg.python.org/cpython/file/b14308524cff/Lib/collections/__init__.py#l378
    # try:
    #     XyAdapter.__module__ = _sys._getframe(1).f_globals.get(
    #         '__name__', '__main__')
    # except (AttributeError, ValueError):
    #     pass

    XyAdapter.__name__ = klass.__name__
    qualname, name = XyAdapter.__qualname__.rsplit('.', 1)
    XyAdapter.__qualname__ = '.'.join((qualname, klass.__name__))

    return XyAdapter


def XyAdapter(klass):
    return XyAdapterFactory(klass)
