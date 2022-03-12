from functools import partial
from scipy import sparse
from sklearn.base import clone
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose._column_transformer import _get_transformer_list
from sklearn.compose import ColumnTransformer as _ColumnTransformer
from sklearn.preprocessing import FunctionTransformer as _FunctionTransformer
from sklearn.utils import _get_column_indices, _safe_indexing
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted, _check_feature_names_in
from typing import Callable
import numpy as np
import pandas as pd


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
    LogisticRegression(random_state=0)
    >>> clf.predict(X[:2, :])
    array([0., 0.])
    >>> clf.predict_proba(X[:2, :])
    array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
           [9.7...e-01, 2.8...e-02, ...e-08]])
    >>> clf.score(train)
    0.97...
    """

    def __init__(self, transformer, target_col=None, ofmt=None):

        self.transformer = transformer

        assert isinstance(target_col, (type(None), str, int, list, tuple))
        if isinstance(target_col, (str, int)):
            target_col = [target_col]
        self.target_col = target_col

        assert isinstance(ofmt, (type(None), str))
        if isinstance(ofmt, str):
            assert ofmt in ['pandas', 'numpy']
        self.ofmt = ofmt

    def _check_ofmt(self, X):

        assert isinstance(X, (pd.DataFrame, pd.Series, np.ndarray))

        ofmt = self.ofmt
        if self.ofmt is None:
            ofmt = 'numpy' if isinstance(X, np.ndarray) else 'pandas'

        return ofmt

    def splitXy(self, Xy, on_error='raise'):
        """Split Xy columns into X and y.

        Parameters
        ----------
        Xy: 2-d array or DataFrame
        on_error: 'raise' or 'ignore'
            Errors when splitting columns are either raised or ignored.

        Returns
        -------
        Returns X and y
        """

        X, y = Xy, None
        if self.target_col is not None:

            features = np.full((Xy.shape[1], ), False)
            try:
                y_cols = _get_column_indices(Xy, self.target_col)
            except ValueError:
                if on_error == 'raise':
                    raise
                else:
                    y_cols = []
            except:
                raise
            features[y_cols] = True

            X = _safe_indexing(Xy, ~features, axis=1)

            y = None
            if y_cols:
                y = _safe_indexing(Xy, features, axis=1)
                y = y.squeeze()  # type: ignore

        return X, y

    def joinXy(self, X, y, ofmt):

        if sparse.issparse(X):
            X = X.todense()

        Xy = X
        if self.target_col is not None:
            if y is not None:
                if any([hasattr(x, 'iloc') for x in (X, y)]):
                    Xy = pd.concat([pd.DataFrame(x) for x in (X, y)], axis=1)
                else:
                    if y.ndim == 1:
                        y = y[:, np.newaxis]
                    Xy = np.hstack((X, y))  # type: ignore

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

            if isinstance(columns, np.ndarray):
                columns = columns.tolist()
            if self.target_col is not None:
                if y is not None:
                    columns.extend(self.target_col)

            if hasattr(Xy, "iloc"):
                Xy.columns = columns  # type: ignore
            else:
                Xy = pd.DataFrame(Xy, columns=columns)

            Xy = Xy.infer_objects()  # type: ignore

        else:

            if hasattr(Xy, "iloc"):
                Xy = Xy.to_numpy()  # type: ignore

        return Xy

    def _call(self, method, X, requires_y=True, join_y=True, reset=True,
              on_error='raise', **params):

        X, y = self.splitXy(X, on_error)
        self._check_feature_names(X, reset=reset)
        self._check_n_features(X, reset=reset)
        ofmt = self._check_ofmt(X)

        if reset:
            self.transformer_ = clone(self.transformer)
        method = getattr(self.transformer_, method)
        if requires_y:
            method = partial(method, X, y, **params)
        else:
            method = partial(method, X, **params)

        if join_y:
            X = method()
            return self.joinXy(X, y, ofmt)
        else:
            return method()

    @if_delegate_has_method("transformer")
    def fit_transform(self, X, _=None, **fit_params):

        return self._call("fit_transform", X, requires_y=True, join_y=True,
                          reset=True, on_error='raise', **fit_params)

    @if_delegate_has_method("transformer")
    def fit(self, X, _=None, **fit_params):

        return self._call("fit", X, requires_y=True, join_y=False, reset=True,
                          on_error='raise', **fit_params)

    @if_delegate_has_method("transformer")
    def transform(self, X):

        check_is_fitted(self)
        return self._call("transform", X, requires_y=False, join_y=True,
                          reset=False, on_error='ignore')

    @if_delegate_has_method("transformer")
    def predict(self, X, **predict_params):

        check_is_fitted(self)
        return self._call("predict", X, requires_y=False, join_y=False,
                          reset=False, on_error='ignore', **predict_params)

    @if_delegate_has_method("transformer")
    def predict_proba(self, X, **predict_proba_params):

        check_is_fitted(self)
        return self._call("predict_proba", X, requires_y=False, join_y=False,
                          reset=False, on_error='ignore',
                          **predict_proba_params)

    @if_delegate_has_method("transformer")
    def score(self, X, _=None, **score_params):

        check_is_fitted(self)
        return self._call("score", X, requires_y=True, join_y=False,
                          reset=False, on_error='ignore', **score_params)


class ColumnTransformer(_ColumnTransformer):
    """Drop-in replacement for Scikit-learn's ColumnTransformer

    Scikit-learn's ColumnTransformer API provides for transformers to specify
    the names of the transformed features' columns using the
    get_feature_names_out API. But it assumes that all the original columns
    that were provided to the transformer was used by the transformer. These
    columns are no longer part of Scikit-learn's 'remainder' API that allows
    for unused columns to either be dropped or passed through. This makes it
    cumbersome to use columns to create new derived columns while also avoiding
    duplicate column names etc.

    This API extension to ColumnTransformer allows for transformers to
    implement a `get_feature_names_used` function that can return which of the
    input features were actually used by it and removed from the list of
    'remainder' columns. The columns not returned by the
    `get_feature_names_used` function will be part of the 'remainder' list and
    can utilize the 'remainder' API functionality even though they were
    provided to one or more transformers. By default, if a transformer does not
    implement this function, it assumes all columns provided to the transformer
    were used by it (which is consistent with the default Scikit-learn
    ColumnTransformer API).

    Parameters
    ----------
        See Scikit-learn's ColumnTransformer

    Attributes
    ----------
        See Scikit-learn's ColumnTransformer

    Examples
    --------

    In this example, ColumnSummer adds all columns in its input to create a
    single final column. The user can name the resulting column. If the summed
    column name matches one of the original input column names, then the column
    whose name matches the final column name is considered 'replaced' by the
    summed column. Else, the summed column is considered a 'new' column.

    In the first example, scikit-learn's FunctionTransformer is subclassed and
    get_feature_names_out and get_feature_names_used are implemented for
    derived class.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn_transformer_extensions import ColumnTransformer
    >>> from sklearn.preprocessing import FunctionTransformer
    >>> class ColumnSummer(FunctionTransformer):
    ...     def __init__(self, name):
    ...         super().__init__(lambda X: np.c_[np.sum(X, axis=1)])
    ...         self.name = name
    ...     def get_feature_names_out(self, _):
    ...         return [self.name]
    ...     def get_feature_names_used(self, input_features):
    ...         return [c for c in input_features if c == self.name]
    >>> ct = ColumnTransformer([("sum1", ColumnSummer(name='sum01'), [0, 1]),
    ...                         ("sum2", ColumnSummer(name='sum23'), [2, 3])],
    ...                        remainder='drop', verbose_feature_names_out=False)
    >>> X = pd.DataFrame(np.array([[0., 1., 2., 2.], [1., 1., 0., 1.]]),
    ...                  columns=['c1', 'c2', 'c3', 'c4'])
    >>> print(X)
        c1   c2   c3   c4
    0  0.0  1.0  2.0  2.0
    1  1.0  1.0  0.0  1.0
    >>> ct.fit_transform(X)
    array([[1., 4.],
           [2., 1.]])
    >>> ct.get_feature_names_out()
    array(['sum01', 'sum23'], dtype='<U5')
    >>> pd.DataFrame(ct.fit_transform(X), columns=ct.get_feature_names_out())
       sum01  sum23
    0    1.0    4.0
    1    2.0    1.0
    >>> ct = ColumnTransformer([("sum1", ColumnSummer(name='sum01'), [0, 1]),
    ...                         ("sum2", ColumnSummer(name='sum23'), [2, 3])],
    ...                        remainder='passthrough', verbose_feature_names_out=False)
    >>> ct.fit_transform(X)
    array([[1., 4., 0., 1., 2., 2.],
           [2., 1., 1., 1., 0., 1.]])
    >>> ct.get_feature_names_out()
    array(['sum01', 'sum23', 'c1', 'c2', 'c3', 'c4'], dtype=object)
    >>> pd.DataFrame(ct.fit_transform(X), columns=ct.get_feature_names_out())
       sum01  sum23   c1   c2   c3   c4
    0    1.0    4.0  0.0  1.0  2.0  2.0
    1    2.0    1.0  1.0  1.0  0.0  1.0
    >>> ct = ColumnTransformer([("sum1", ColumnSummer(name='c1'), [0, 1]),
    ...                         ("sum2", ColumnSummer(name='sum23'), [2, 3])],
    ...                        remainder='passthrough', verbose_feature_names_out=False)
    >>> pd.DataFrame(ct.fit_transform(X), columns=ct.get_feature_names_out())
        c1  sum23   c2   c3   c4
    0  1.0    4.0  1.0  2.0  2.0
    1  2.0    1.0  1.0  0.0  1.0

    In the second example, ColumnSummer is implemented by subclassing the
    extension to FunctionTransformer that is implemented in this package (see
    `FunctionTransformer`).

    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn_transformer_extensions import ColumnTransformer
    >>> from sklearn_transformer_extensions import FunctionTransformer
    >>> class ColumnSummer(FunctionTransformer):
    ...     def __init__(self, col_name):
    ...         super().__init__(lambda X: np.c_[np.sum(X, axis=1)], col_name=col_name)
    >>> ct = ColumnTransformer([("sum1", ColumnSummer(col_name='c1'), [0, 1]),
    ...                         ("sum2", ColumnSummer(col_name='sum23'), [2, 3])],
    ...                        remainder='passthrough', verbose_feature_names_out=False)
    >>> X = pd.DataFrame(np.array([[0., 1., 2., 2.], [1., 1., 0., 1.]]),
    ...                  columns=['c1', 'c2', 'c3', 'c4'])
    >>> pd.DataFrame(ct.fit_transform(X), columns=ct.get_feature_names_out())
        c1  sum23   c2   c3   c4
    0  1.0    4.0  1.0  2.0  2.0
    1  2.0    1.0  1.0  0.0  1.0
    """

    def _validate_remainder(self, X):
        """
        Validates ``remainder`` and defines ``_remainder`` targeting
        the remaining columns.
        """
        # yapf: disable
        is_transformer = (
            hasattr(self.remainder, "fit") or hasattr(self.remainder, "fit_transform")
        ) and hasattr(self.remainder, "transform")
        if self.remainder not in ("drop", "passthrough") and not is_transformer:
            raise ValueError(
                "The remainder keyword needs to be one of 'drop', "
                "'passthrough', or estimator. '%s' was passed instead"
                % self.remainder
            )
        # yapf: enable

        check_is_fitted(self)
        input_features = _check_feature_names_in(self)
        input_features_inv = pd.DataFrame(columns=input_features)

        self._n_features = X.shape[1]
        used_cols = set()
        for name, trans, _ in self.transformers:
            columns = self._transformer_to_input_indices[name]
            columns = _safe_indexing(input_features, columns)

            used_cols_ = columns
            if hasattr(trans, "get_feature_names_used"):
                used_cols_ = trans.get_feature_names_used(columns)
            if isinstance(used_cols, np.ndarray):
                used_cols_ = used_cols.tolist()
            
            used_cols_ = _get_column_indices(input_features_inv, used_cols_)
            used_cols.update(used_cols_)

        cols = set(_get_column_indices(X, sorted(used_cols)))
        remaining = sorted(set(range(self._n_features)) - cols)
        self._remainder = ("remainder", self.remainder, remaining)
        self._transformer_to_input_indices["remainder"] = remaining


def make_column_transformer(
    *transformers,
    remainder="drop",
    sparse_threshold=0.3,
    n_jobs=None,
    verbose=False,
    verbose_feature_names_out=True,
):
    """Drop-in replacement for Scikit-learn's make_column_transformer"""

    transformer_list = _get_transformer_list(transformers)
    return ColumnTransformer(
        transformer_list,
        n_jobs=n_jobs,
        remainder=remainder,
        sparse_threshold=sparse_threshold,
        verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out,
    )


class FunctionTransformer(_FunctionTransformer):
    """Drop-in replacement for Scikit-learn's FunctionTransformer

    The extension provides for taking a fit_fn callable as argument that gets
    called during the fit stage. The returned object from the fit_fn is passed
    to FunctionTransformer's `func` function that is called during the
    transform stage using FunctionTransformer's kw_args API mechanism.

    The user can specify what the names of the transformed columns should be.
    This function also provides a `get_feature_names_used` function that
    conforms ColumnTransformer's API extension.

    Parameters
    ----------
    func: Callable, required
        See FunctionTransformer for more details. Exposed here as they are
        positional arguments in FunctionTransformer and sklearn's transformer
        api requires them to be explicitly when subclassing.
    inverse_func: Callable, default=None
        See FunctionTransformer for more details. Exposed here as they are
        positional arguments in FunctionTransformer and sklearn's transformer
        api requires them to be explicitly when subclassing.
    fit_fn: Callable, default=None
        Function called during the fit stage. The returned object is passed
        through FunctionTransformer's kw_args api to the `func` function that
        is called during transform stage. If the returned object is a dict,
        then the returned object is passed as keyword arguments to `func`. If
        the returned object is not a dict, then then returned object is passed
        to `func` as the `params` keyword argument.
    col_name: str, list, tuple, Callable, None, default=None
        Used to set the names of the transformed features returned by `func`.
        If col_name is a string, list or tuple, they are used to set the column
        names of the transformed features. If col_name is a callable, the
        col_name function is called with the names of the input columns as
        argument. The returned list from the function is used to set the column
        names of the transformed features. Of the original input columns, only
        those columns that are part of the transformed column list are
        considered used and returned by the `get_feature_names_used` function.
        See the `get_feature_names_used` ColumnTransformer API extension
        documentation for more details.

    Attributes
    -----------
    None

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn_transformer_extensions import FunctionTransformer
    >>> fit_fn = lambda X, _: dict(p=1)
    >>> transform_fn = lambda X, p: X + p
    >>> ft = FunctionTransformer(transform_fn, fit_fn=fit_fn, validate=True)
    >>> X = pd.DataFrame(np.array([[0., 1., 2., 2.], [1., 1., 0., 1.]]),
    ...                  columns=['c1', 'c2', 'c3', 'c4'])
    >>> print(X)
        c1   c2   c3   c4
    0  0.0  1.0  2.0  2.0
    1  1.0  1.0  0.0  1.0
    >>> ft.fit_transform(X)
    array([[1., 2., 3., 3.],
           [2., 2., 1., 2.]])
    >>> ft.get_feature_names_out()
    ['c1', 'c2', 'c3', 'c4']
    >>> fit_fn = lambda X, _: dict(p=X.sum().to_numpy())
    >>> transform_fn = lambda X, p: X + p
    >>> ft = FunctionTransformer(transform_fn, fit_fn=fit_fn, validate=True)
    >>> print(X)
        c1   c2   c3   c4
    0  0.0  1.0  2.0  2.0
    1  1.0  1.0  0.0  1.0
    >>> ft.fit_transform(X)
    array([[1., 3., 4., 5.],
           [2., 3., 2., 4.]])
    >>> ft.get_feature_names_out()
    ['c1', 'c2', 'c3', 'c4']
    >>> fit_fn = lambda X, _: dict(p=X.sum().to_numpy())
    >>> transform_fn = lambda X, p: X / p
    >>> col_name_fn = lambda cols: [c + '_prob' for c in cols]
    >>> ft = FunctionTransformer(transform_fn, fit_fn=fit_fn, 
    ...                          col_name=col_name_fn, validate=True)
    >>> ft.fit_transform(X)
    array([[0.        , 0.5       , 1.        , 0.66666667],
           [1.        , 0.5       , 0.        , 0.33333333]])
    >>> ft.get_feature_names_out()
    ['c1_prob', 'c2_prob', 'c3_prob', 'c4_prob']
    """ 

    def __init__(self, func, inverse_func=None, *, fit_fn=None, col_name=None,
                 **kwargs):

        super().__init__(func, inverse_func, **kwargs)

        assert isinstance(fit_fn, (type(None), Callable))
        self.fit_fn = fit_fn

        assert isinstance(col_name, (type(None), str, list, tuple, Callable))
        if isinstance(col_name, str):
            col_name = [col_name]
        elif isinstance(col_name, tuple):
            col_name = list(col_name)
        self.col_name = col_name

    def fit(self, X, y=None, **fit_params):
        super().fit(X, y, **fit_params)
        if self.fit_fn is not None:
            self.kw_args = self.kw_args if self.kw_args else {}
            ret = self.fit_fn(X, y)
            if not isinstance(ret, dict):
                ret = dict(params=ret)
            self.kw_args.update(ret)

        return self

    def get_feature_names_out(self, input_features=None) -> list:

        check_is_fitted(self)
        input_features = _check_feature_names_in(self, input_features).tolist()

        if isinstance(self.col_name, list):
            return self.col_name
        elif isinstance(self.col_name, Callable):
            return self.col_name(input_features)
        else:
            return input_features

    def get_feature_names_used(self, input_features) -> list:

        check_is_fitted(self)
        input_features = _check_feature_names_in(self, input_features).tolist()

        transformed_features = self.get_feature_names_out(input_features)
        transformed_features = set(transformed_features)
        used_features = [c for c in input_features if c in transformed_features]

        return used_features
