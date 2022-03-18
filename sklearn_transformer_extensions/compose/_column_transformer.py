from sklearn.compose._column_transformer import _get_transformer_list
from sklearn.utils import _get_column_indices, _safe_indexing
from sklearn.utils.validation import check_is_fitted, _check_feature_names_in
from sklearn.compose import ColumnTransformer as _ColumnTransformer


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
    >>> from sklearn_transformer_extensions.compose import ColumnTransformer
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
    >>> from sklearn_transformer_extensions.compose import ColumnTransformer
    >>> from sklearn_transformer_extensions.preprocessing import FunctionTransformer
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

        self._n_features = X.shape[1]
        used_cols = set()
        for name, trans, _ in self.transformers:
            columns = self._transformer_to_input_indices[name]
            columns = _safe_indexing(input_features, columns)

            used_cols_ = columns
            if hasattr(trans, "get_feature_names_used"):
                used_cols_ = trans.get_feature_names_used(columns)

            used_cols.update(used_cols_)

        used_cols_idx = set(
            [i for i, k in enumerate(input_features) if k in used_cols])
        remaining = sorted(set(range(self._n_features)) - used_cols_idx)
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
