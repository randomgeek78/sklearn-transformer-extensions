from sklearn.pipeline import Pipeline as _Pipeline, _name_estimators
from .xydata import XyData
from .xyadapter import XyAdapter, XyAdapterBase

from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion as _FeatureUnion


def _convert_xy(trans):

    if isinstance(trans, (type(None), XyAdapterBase)) or trans == 'passthrough':
        return trans

    init_params = {}
    if hasattr(trans, "get_params"):
        init_params = trans.get_params(False)
    newtrans = trans
    # If the class has this attribute, then don't wrap around XyAdapter
    if not hasattr(trans.__class__,
                   "__xydata__") or trans.__class__.__xydata__ is False:
        newtrans = XyAdapter(trans.__class__)(**init_params)
        newtrans.__dict__ = trans.__dict__
    return newtrans


class Pipeline(_Pipeline):
    """Lightweight specialization of `sklearn.pipeline.Pipeline` that
    automatically wraps all steps provided to the pipeline with the XyAdapter
    (when called with xy_adapter=True). In all other aspects, it is a drop-in
    replacement to sklearn's Pipeline. When xy_adapter = True, classes can
    prevent XyAdapter wrapping by creating and setting the class attribute
    __xydata__=True.

    Parameters
    ----------
    See sklearn.pipeline.Pipeline

    Attributes
    ----------
    See sklearn.pipeline.Pipeline

    Examples
    --------
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn_transformer_extensions import XyAdapter, XyData
    >>> from sklearn_transformer_extensions.pipeline import (
    ...     Pipeline as PipelineXy)
    >>> from sklearn.pipeline import Pipeline

    >>> JUNK_FOOD_DOCS = (
    ...     "the pizza pizza beer copyright",
    ...     "the pizza burger beer copyright",
    ...     "the the pizza beer beer copyright",
    ...     "the burger beer beer copyright",
    ...     "the coke burger coke copyright",
    ...     "the coke burger burger",
    ... )

    >>> X = JUNK_FOOD_DOCS
    >>> y = ["pizza" in x for x in JUNK_FOOD_DOCS]
    >>> Xy = XyData(X, y)
    >>> pipe = Pipeline(steps=[
    ...     ("vect", XyAdapter(CountVectorizer)()),
    ...     ("clf", XyAdapter(LogisticRegression)()),
    ... ])
    >>> pipe.fit(Xy)
    Pipeline(steps=[('vect', CountVectorizer()), ('clf', LogisticRegression())])
    >>> pipe.score(Xy)
    1.0
    >>> pipe[0].get_feature_names_out()
    array(['beer', 'burger', 'coke', 'copyright', 'pizza', 'the'],
          dtype=object)

    This can also be accomplished using the extended Pipeline class implemented
    in this module. In this case, individual steps don't need to be wrapped
    with XyAdapter.
    >>> pipe = PipelineXy(steps=[
    ...     ("vect", CountVectorizer()),
    ...     ("clf", LogisticRegression()),
    ... ], xy_adapter=True)
    >>> pipe.fit(Xy)
    Pipeline(steps=[('vect', CountVectorizer()), ('clf', LogisticRegression())],
             xy_adapter=True)
    >>> pipe[0].get_feature_names_out()
    array(['beer', 'burger', 'coke', 'copyright', 'pizza', 'the'],
          dtype=object)

    """

    def __init__(self, steps, *, memory=None, verbose=False, xy_adapter=False):

        if type(steps) == tuple:
            raise TypeError(
                "Providing 'steps' as a tuple is not supported. Please"
                " provide 'steps' as a list.")

        assert type(xy_adapter) == bool
        self.xy_adapter = xy_adapter

        if self.xy_adapter:
            for i, (name, trans) in enumerate(steps):
                steps[i] = (name, _convert_xy(trans))

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
                if self.xy_adapter:
                    trans = _convert_xy(params.pop(name))
                else:
                    trans = params.pop(name)
                self._replace_estimator(attr, name, trans)
        # 3. Step parameters and other initialisation arguments
        BaseEstimator.set_params(self, **params)
        return self


def make_pipeline(*steps, memory=None, verbose=False, xy_adapter=False):
    """See sklearn.pipeline.make_pipeline

    Calls sklearn_transformer_extensions.pipeline.Pipeline

    Examples
    --------
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn_transformer_extensions.pipeline import make_pipeline
    >>> make_pipeline(StandardScaler(), GaussianNB(priors=None))
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('gaussiannb', GaussianNB())])
    """
    return Pipeline(_name_estimators(steps), memory=memory, verbose=verbose,
                    xy_adapter=xy_adapter)


class FeatureUnion(_FeatureUnion):

    __xydata__ = True

    def _hstack(self, Xs):
        input_types = list(set([type(X) for X in Xs]))
        assert len(input_types) == 1
        input_type = input_types[0]
        if input_type == XyData:
            Xs, ys = zip(*Xs)
            X = super(FeatureUnion, self)._hstack(Xs)
            return XyData(X, ys[0])
        return super(FeatureUnion, self)._hstack(Xs)


def make_union(*transformers, n_jobs=None, verbose=False):
    return FeatureUnion(_name_estimators(transformers), n_jobs=n_jobs,
                        verbose=verbose)
