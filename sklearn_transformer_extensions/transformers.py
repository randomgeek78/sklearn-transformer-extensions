from scipy import sparse
from sklearn.base import clone
from sklearn.base import _OneToOneFeatureMixin
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
    """A adapter class for  pipeline that keeps the features (X) and labels (y)
    together in the same datastructure.

    Scikit-learn API handle features (X) and labels (y) independently. The main
    downside is that it becomes very difficult to handle outliers during the
    fitting stage as part of a unified pipeline. In these cases, it is easier
    to keep both X and y as part of the same datastructure. But then, it cannot
    be passed directly to fitting algorithms in scikit-learn since they expect
    X and y to be provided separately (for supervised learning algorithms).

    This adapter class works with a single X object that contains both the
    features and labels. It transparently separates out the X and y before
    passing it off to the underlying transformer or estimator.


    Parameters
    ----------
    transformer: a single estimator/transformer instance, required
        The estimator or group of estimators to be cloned.
    target_col : list[str], list[int], str, int or None, default=None
        The columns to pry away from the input X that correspond to the labels
        (y) before calling the underlying transformer.

    Returns
    -------
    transformer_: a fitted instance of the estimator/transformer
    """

    def __init__(self, transformer, target_col=None):

        self.transformer = transformer
        if not isinstance(target_col, (type(None), list, tuple)):
            target_col = [target_col]
        self.target_col = target_col

    def splitXy(self, Xy):

        X, y = Xy, None
        if self.target_col is not None:
            features = np.full((Xy.shape[1], ), False)
            try:
                y_cols = _get_column_indices(Xy, self.target_col)
            except:
                y_cols = []
            features[y_cols] = True

            y = _safe_indexing(Xy, features, axis=1)
            X = _safe_indexing(Xy, ~features, axis=1)
            if y.shape[1] == 0:  # type: ignore
                y = None

        return X, y

    def joinXy(self, X, y, is_pandas_df):

        if sparse.issparse(X):
            X = X.todense()

        Xy = X
        if self.target_col is not None:
            if y is not None:
                Xy = np.hstack((X, y))  # type: ignore

        if is_pandas_df:

            columns = None
            for attr in ["get_feature_names_out", "get_feature_names"]:
                try:
                    columns = getattr(self.transformer_, attr)()
                    break
                except:
                    pass
            if columns is None:
                columns = self.feature_names_in_

            if isinstance(columns, np.ndarray):
                columns = columns.tolist()
            if self.target_col is not None:
                if y is not None:
                    columns.extend(self.target_col)
            Xy = pd.DataFrame(Xy, columns=columns).infer_objects()

        return Xy

    @if_delegate_has_method("transformer")
    def fit_transform(self, Xy, *_):

        X, y = self.splitXy(Xy)
        self._check_feature_names(X, reset=True)

        y_ = None
        if isinstance(y, pd.DataFrame):
            y_ = y.squeeze()
        self.transformer_ = clone(self.transformer)
        Xt = self.transformer_.fit_transform(X, y_)  # type: ignore
        Xty = self.joinXy(Xt, y, isinstance(Xy, pd.DataFrame))

        return Xty

    @if_delegate_has_method("transformer")
    def fit(self, Xy, *_):

        X, y = self.splitXy(Xy)
        self._check_feature_names(X, reset=True)

        y_ = None
        if isinstance(y, pd.DataFrame):
            y_ = y.squeeze()
        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(X, y_)  # type: ignore

        return self

    @if_delegate_has_method("transformer")
    def transform(self, Xy):

        X, y = self.splitXy(Xy)
        self._check_feature_names(X, reset=False)
        check_is_fitted(self)

        Xt = self.transformer_.transform(X)  # type: ignore
        Xty = self.joinXy(Xt, y, isinstance(Xy, pd.DataFrame))

        return Xty

    @if_delegate_has_method("transformer")
    def predict(self, Xy, **predict_params):

        X, _ = self.splitXy(Xy)
        self._check_feature_names(X, reset=False)
        check_is_fitted(self)

        y_pred = self.transformer_.predict(X, **predict_params)  # type: ignore

        if self.target_col is not None:
            y_pred = pd.DataFrame(y_pred, columns=self.target_col).squeeze()

        return y_pred

    @if_delegate_has_method("transformer")
    def predict_proba(self, Xy, **predict_proba_params):

        X, _ = self.splitXy(Xy)
        self._check_feature_names(X, reset=False)
        check_is_fitted(self)

        # yapf: disable
        y_pred = self.transformer_.predict_proba(X, **predict_proba_params) # type: ignore
        # yapf: enable

        if self.target_col is not None:
            y_pred = pd.DataFrame(
                y_pred,
                columns=self.transformer_.classes_).squeeze()  # type: ignore

        return y_pred


class ColumnTransformer(_ColumnTransformer):
    """Extension to allow transformers to indicate to ColumnTransformer which
    columns they actually use so the rest of the columns are treated by
    ColumnTransformer as remainder columns. 

    By default, all columns that were supplied to one or more transformers are
    considered by ColumnTransformer to have been used.

    In order to indicate which columns, transformers need to implement a
    ``get_features_used`` function that takes in a list of original columns
    supplied to the transformer and return a list of columns that were actually
    used by the transformer. If this function is not implemented by the
    transformer, then ColumnTransformer will fallback to the default behaviour
    of assuming that all the supplied columns were used by the transformer.
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

        self._n_features = X.shape[1]
        cols = set()
        for (_, trans, _), columns in zip(self.transformers, self._columns):
            if hasattr(trans, "get_features_used"):
                cols.update(trans.get_features_used(columns))
            else:
                cols.update(columns)
        cols = set(_get_column_indices(X, sorted(cols)))
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
    """A drop-in replacement for Scikit-learn's FunctionTransformer

    In addition to taking a function that is used in the transform stage, it
    also takes in a function to be used in the fit stage. In order to play well
    with pandas DataFrames and the modified ColumnTransformer, it also provides
    the ability to change the name of the columns.
    """

    def __init__(self, func, inverse_func=None, *, fit_fn=None, col_name=None, **kwargs):

        super().__init__(func, inverse_func, **kwargs)

        assert isinstance(fit_fn, (type(None), Callable))
        self.fit_fn = fit_fn

        assert isinstance(col_name, (type(None), str, list, tuple, Callable))
        if isinstance(col_name, str):
            col_name = [col_name]
        elif isinstance(col_name, tuple):
            col_name = list(col_name)
        self.col_name = col_name

        self.params = None

    def fit(self, X, y=None, **fit_params):
        super().fit(X, y, **fit_params)
        if self.fit_fn is not None:
            self.kw_args = self.kw_args if self.kw_args else {}
            ret = self.fit_fn(X, y)
            if not isinstance(ret, dict):
                ret = dict(params=ret)
            self.kw_args.update(ret)
        return self

    def get_feature_names_out(self, columns=None) -> list:
        if isinstance(self.col_name, list):
            return self.col_name
        elif isinstance(self.col_name, Callable):
            return self.col_name(columns)
        else:
            return _check_feature_names_in(self, columns).tolist()

    def get_features_used(self, columns) -> list:
        final_columns = set(self.get_feature_names_out(columns))
        used_columns = [c for c in columns if c in final_columns]
        return used_columns
