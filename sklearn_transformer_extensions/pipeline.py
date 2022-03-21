from sklearn.pipeline import Pipeline as _Pipeline, _name_estimators
from sklearn_transformer_extensions import XyAdapter, XyAdapterBase
from sklearn.base import BaseEstimator


def _convert_xy(trans):

    if isinstance(trans, (type(None), XyAdapterBase)) or trans == 'passthrough':
        return trans

    init_params = {}
    if hasattr(trans, "get_params"):
        init_params = trans.get_params(False)
    xytrans = XyAdapter(trans.__class__)(**init_params)
    xytrans.__dict__ = trans.__dict__
    return xytrans


class Pipeline(_Pipeline):
    """Lightweight specialization of `sklearn.pipeline.Pipeline` that
    automatically wraps all steps provided to the pipeline with the XyAdapter.
    In all other aspects, it is a drop-in replacement to sklearn's Pipeline.

    Parameters
    ----------
    See sklearn.pipeline.Pipeline

    Attributes
    ----------
    See sklearn.pipeline.Pipeline

    Examples
    --------



    """

    def __init__(self, steps, *, memory=None, verbose=False):

        if type(steps) == tuple:
            raise TypeError(
                "Providing 'steps' as a tuple is not supported. Please"
                " provide 'steps' as a list.")

        for i, (name, trans) in enumerate(steps):
            steps[i] = (name, _convert_xy(trans))
            set_params = {}
            if hasattr(trans, "get_params"):
                set_params = {
                    k: v
                    for k, v in trans.get_params(True).items() if '__' in k
                }
            if set_params:
                steps[i][1].set_params(**set_params)

        super().__init__(steps, memory=memory, verbose=verbose)

    def _set_params(self, attr, **params):
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. Step replacement
        items = getattr(self, attr)
        names = []
        if items:
            names, _ = zip(*items)
        for name in list(params.keys()):
            if "__" not in name and name in names:
                trans = _convert_xy(params.pop(name))
                self._replace_estimator(attr, name, trans)
        # 3. Step parameters and other initialisation arguments
        BaseEstimator.set_params(self, **params)
        return self


def make_pipeline(*steps, memory=None, verbose=False):
    """See sklearn.pipeline.make_pipeline

    Calls sklearn_transformer_extensions.pipeline.Pipeline

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('gaussiannb', GaussianNB())])
    """
    return Pipeline(_name_estimators(steps), memory=memory, verbose=verbose)
