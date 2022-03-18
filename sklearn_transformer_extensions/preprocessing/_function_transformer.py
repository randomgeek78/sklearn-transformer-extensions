from sklearn.utils.validation import check_is_fitted, _check_feature_names_in
from typing import Callable
from sklearn.preprocessing import FunctionTransformer as _FunctionTransformer


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
    >>> from sklearn_transformer_extensions.preprocessing import FunctionTransformer
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

    def __init__(self, func=None, inverse_func=None, *, fit_fn=None,
                 col_name=None, **kwargs):

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
