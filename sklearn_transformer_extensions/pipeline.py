from sklearn.base import clone
from sklearn.pipeline import Pipeline as _Pipeline, _name_estimators
from sklearn_transformer_extensions import XyAdapter
from typing import Iterable
from sklearn.utils.validation import check_memory
from sklearn.pipeline import _fit_transform_one
from sklearn.utils import _print_elapsed_time


class Pipeline(_Pipeline):

    def __init__(self, steps, target_col=None, *, ofmt=None, memory=None,
                 verbose=False):

        if type(steps) == tuple:
            raise TypeError(
                "Providing 'steps' as a tuple is not supported. Please"
                " provide 'steps' as a list.")

        self.target_col = target_col
        self.ofmt = ofmt
        self.steps_ = steps

        super().__init__(steps, memory=memory, verbose=verbose)

    @property
    def steps(self):
        return self.steps__

    @steps.setter
    def steps(self, steps):
        # Shallow copy
        steps = list(steps)
        for i, (name, trans) in enumerate(steps):
            if hasattr(trans, "transformer"):
                trans = trans.transformer
            if trans in ['passthrough', None]:
                continue
            steps[i] = (name, XyAdapter(trans, self.target_col, ofmt=self.ofmt))

        self.steps__ = steps
        return self

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)

        for name, trans in params['steps']:
            if trans in ['passthrough', None]:
                continue
            if name in params:
                params[name] = trans.transformer

        for i, (name, trans) in enumerate(params['steps']):
            if trans in ['passthrough', None]:
                self.steps_[i] = (name, trans)
            else:
                self.steps_[i] = (name, trans.transformer)
        params['steps'] = self.steps_

        return params

    def __getitem__(self, ind):
        """Returns a sub-pipeline or a single estimator in the pipeline

        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.
        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            params = self.get_params(deep=False)
            steps = params.pop('steps')
            return self.__class__(steps[ind], **params)
        try:
            name, est = self.steps[ind]
        except TypeError:
            # Not an int, try get step by name
            return self.named_steps[ind]
        return est

    def _fit(self, X, y=None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for (step_idx, name,
             transformer) in self._iter(with_final=False,
                                        filter_passthrough=False):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline",
                                         self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location"):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    transformer = transformer.transformer
                    cloned_transformer = XyAdapter(clone(transformer),
                                                   self.target_col,
                                                   ofmt=self.ofmt)
            elif hasattr(memory, "cachedir"):
                # joblib < 0.11
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    transformer = transformer.transformer
                    cloned_transformer = XyAdapter(clone(transformer),
                                                   self.target_col,
                                                   ofmt=self.ofmt)
            else:
                transformer = transformer.transformer
                cloned_transformer = XyAdapter(clone(transformer),
                                               self.target_col, ofmt=self.ofmt)

            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(  # type: ignore
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",  # type: ignore
                message=self._log_message(step_idx),  # type: ignore
                **fit_params_steps[name],
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X


def make_pipeline(*steps, target_col=None, ofmt=None, memory=None,
                  verbose=False):
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
    return Pipeline(_name_estimators(steps), target_col, ofmt=ofmt,
                    memory=memory, verbose=verbose)
