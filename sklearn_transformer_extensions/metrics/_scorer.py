from sklearn.utils.validation import check_is_fitted
from sklearn_transformer_extensions import XyData
from sklearn.metrics import make_scorer as _make_scorer


def make_scorer(*args, **kwargs):

    _scorer = _make_scorer(*args, **kwargs)

    def scorer(estimator, X, y_true=None, sample_weight=None):

        check_is_fitted(estimator)

        if hasattr(estimator, "steps") and hasattr(estimator, "_iter"):
            for _, name, transform in estimator._iter(with_final=False):
                X = transform.transform(X)
            if type(X) == XyData:
                X, y_true = X
            estimator = estimator[-1]

        return _scorer(estimator, X, y_true, sample_weight)

    return scorer
