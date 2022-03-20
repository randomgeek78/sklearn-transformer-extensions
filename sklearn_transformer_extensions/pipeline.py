from sklearn.pipeline import Pipeline as _Pipeline, _name_estimators
from sklearn_transformer_extensions import XyAdapter, XyAdapterBase
from sklearn.base import BaseEstimator


def convert_xy(trans):

    if isinstance(trans, (type(None), XyAdapterBase)) or trans == 'passthrough':
        return trans

    init_params = {}
    if hasattr(trans, "get_params"):
        init_params = trans.get_params(False)
    xytrans = XyAdapter(trans.__class__)(**init_params)
    xytrans.__dict__ = trans.__dict__
    return xytrans


class Pipeline(_Pipeline):

    def __init__(self, steps, *, memory=None, verbose=False):

        if type(steps) == tuple:
            raise TypeError(
                "Providing 'steps' as a tuple is not supported. Please"
                " provide 'steps' as a list.")

        for i, (name, trans) in enumerate(steps):
            steps[i] = (name, convert_xy(trans))
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
                trans = convert_xy(params.pop(name))
                self._replace_estimator(attr, name, trans)
        # 3. Step parameters and other initialisation arguments
        BaseEstimator.set_params(self, **params)
        return self


def make_pipeline(*steps, memory=None, verbose=False):
    """Construct a :class:`Pipeline` from the given estimators.

    This is a shorthand for the :class:`Pipeline` constructor; it does not
    require, and does not permit, naming the estimators. Instead, their names
    will be set to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of Estimator objects
        List of the scikit-learn estimators that are chained together.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Returns
    -------
    p : Pipeline
        Returns a scikit-learn :class:`Pipeline` object.

    See Also
    --------
    Pipeline : Class for creating a pipeline of transforms with a final
        estimator.

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
