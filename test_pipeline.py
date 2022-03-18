"""
Test the pipeline module.
"""
from tempfile import mkdtemp
import shutil
import time
import re
import itertools

import pytest
import numpy as np
from scipy import sparse
import joblib

from sklearn.utils.fixes import parse_version
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_equal,
    assert_array_almost_equal,
    MinimalClassifier,
    MinimalRegressor,
    MinimalTransformer,
)
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone, is_classifier, BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, make_union
from sklearn_transformer_extensions.pipeline import Pipeline, make_pipeline
from sklearn.pipeline import Pipeline as _Pipeline, make_pipeline as _make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.dummy import DummyRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

from sklearn_transformer_extensions import XyAdapter, XyData
from functools import partial

iris = load_iris()

JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)


class NoFit:
    """Small class to test parameter dispatching."""

    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class NoTrans(NoFit):

    def fit(self, X, y):
        return self

    def get_params(self, deep=False):
        return {"a": self.a, "b": self.b}

    def set_params(self, **params):
        self.a = params["a"]
        return self


class NoInvTransf(NoTrans):

    def transform(self, X):
        return X


class Transf(NoInvTransf):

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class TransfFitParams(Transf):

    def fit(self, X, y, **fit_params):
        self.fit_params = fit_params
        return self


class Mult(BaseEstimator):

    def __init__(self, mult=1):
        self.mult = mult

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.asarray(X) * self.mult

    def inverse_transform(self, X):
        return np.asarray(X) / self.mult

    def predict(self, X):
        return (np.asarray(X) * self.mult).sum(axis=1)

    predict_proba = predict_log_proba = decision_function = predict

    def score(self, X, y=None):
        return np.sum(X)


class FitParamT(BaseEstimator):
    """Mock classifier"""

    def __init__(self):
        self.successful = False

    def fit(self, X, y, should_succeed=False):
        self.successful = should_succeed

    def predict(self, X):
        return self.successful

    def fit_predict(self, X, y, should_succeed=False):
        self.fit(X, y, should_succeed=should_succeed)
        return self.predict(X)

    def score(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            X = X * sample_weight
        return np.sum(X)


class DummyTransf(Transf):
    """Transformer which store the column means"""

    def fit(self, X, y):
        self.means_ = np.mean(X, axis=0)
        # store timestamp to figure out whether the result of 'fit' has been
        # cached or not
        self.timestamp_ = time.time()
        return self


class DummyEstimatorParams(BaseEstimator):
    """Mock classifier that takes params on predict"""

    def fit(self, X, y):
        return self

    def predict(self, X, got_attribute=False):
        self.got_attribute = got_attribute
        return self

    def predict_proba(self, X, got_attribute=False):
        self.got_attribute = got_attribute
        return self

    def predict_log_proba(self, X, got_attribute=False):
        self.got_attribute = got_attribute
        return self


@pytest.mark.parametrize("Pipeline", [Pipeline, _Pipeline])
def test_pipeline_init(Pipeline):
    # Test the various init parameters of the pipeline.
    with pytest.raises(TypeError):
        Pipeline()

    # Check that we can't instantiate pipelines with objects without fit
    # method
    msg = ("Last step of Pipeline should implement fit "
           "or be the string 'passthrough'"
           ".*NoFit.*")
    with pytest.raises(TypeError, match=msg):
        Pipeline([("clf", NoFit())])

    # Smoke test with only an estimator
    clf = NoTrans()
    pipe = Pipeline([("svc", clf)])
    assert pipe.get_params(deep=True) == dict(svc__a=None, svc__b=None, svc=clf,
                                              **pipe.get_params(deep=False))

    # Check that params are set
    pipe.set_params(svc__a=0.1)
    assert clf.a == 0.1
    assert clf.b is None
    # Smoke test the repr:
    repr(pipe)

    # Test with two objects
    clf = SVC()
    filter1 = SelectKBest(f_classif)
    pipe = Pipeline([("anova", filter1), ("svc", clf)])

    # Check that estimators are not cloned on pipeline construction
    if Pipeline is _Pipeline:
        assert pipe.named_steps["anova"] is filter1
        assert pipe.named_steps["svc"] is clf
    else:
        assert pipe.named_steps["anova"].transformer is filter1
        assert pipe.named_steps["svc"].transformer is clf

    # Check that we can't instantiate with non-transformers on the way
    # Note that NoTrans implements fit, but not transform
    msg = "All intermediate steps should be transformers.*\\bNoTrans\\b.*"
    with pytest.raises(TypeError, match=msg):
        Pipeline([("t", NoTrans()), ("svc", clf)])

    # Check that params are set
    pipe.set_params(svc__C=0.1)
    assert clf.C == 0.1
    # Smoke test the repr:
    repr(pipe)

    # Check that params are not set when naming them wrong
    msg = "Invalid parameter C for estimator SelectKBest"
    with pytest.raises(ValueError, match=msg):
        pipe.set_params(anova__C=0.1)

    # Test clone
    with pytest.warns(None):
        pipe2 = clone(pipe)
    assert not pipe.named_steps["svc"] is pipe2.named_steps["svc"]

    # Check that apart from estimators, the parameters are the same
    params = pipe.get_params(deep=True)
    params2 = pipe2.get_params(deep=True)

    for x in pipe.get_params(deep=False):
        params.pop(x)

    for x in pipe2.get_params(deep=False):
        params2.pop(x)

    # Remove estimators that where copied
    params.pop("svc")
    params.pop("anova")
    params2.pop("svc")
    params2.pop("anova")
    assert params == params2


@pytest.mark.parametrize("Pipeline", [_Pipeline, Pipeline])
def test_pipeline_init_tuple(Pipeline):
    # Pipeline accepts steps as tuple
    if Pipeline is _Pipeline:
        X = np.array([[1, 2]])
        pipe = Pipeline((("transf", Transf()), ("clf", FitParamT())))
        pipe.fit(X, y=None)
        pipe.score(X)

        pipe.set_params(transf="passthrough")
        pipe.fit(X, y=None)
        pipe.score(X)
    else:
        msg = "Providing 'steps' as a tuple is not supported. Please provide"\
              " 'steps' as a list."
        with pytest.raises(TypeError, match=msg):
            pipe = Pipeline((("transf", Transf()), ("clf", FitParamT())))


@pytest.mark.parametrize("Pipeline,Xyfn", [
    (_Pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=None), lambda X, y: XyData(X, y)),
])
def test_pipeline_methods_anova(Pipeline, Xyfn):
    # Test the various methods of the pipeline (anova).
    X = iris.data
    y = iris.target
    X = Xyfn(X, y)
    # Test with Anova + LogisticRegression
    clf = LogisticRegression()
    filter1 = SelectKBest(f_classif, k=2)
    pipe = Pipeline([("anova", filter1), ("logistic", clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)

    pipe.score(X, y)


@pytest.mark.parametrize("Pipeline", [_Pipeline, Pipeline])
def test_pipeline_fit_params(Pipeline):
    # Test that the pipeline can take fit parameters
    pipe = Pipeline([("transf", Transf()), ("clf", FitParamT())])
    pipe.fit(X=None, y=None, clf__should_succeed=True)
    # classifier should return True
    assert pipe.predict(None)
    # and transformer params should not be changed
    assert pipe.named_steps["transf"].a is None
    assert pipe.named_steps["transf"].b is None
    # invalid parameters should raise an error message

    msg = re.escape("fit() got an unexpected keyword argument 'bad'")
    with pytest.raises(TypeError, match=msg):
        pipe.fit(None, None, clf__bad=True)


@pytest.mark.parametrize("Pipeline,Xyfn", [
    (_Pipeline, lambda X: X),
    (partial(Pipeline, target_col=None), lambda X: X),
])
def test_pipeline_sample_weight_supported(Pipeline, Xyfn):
    # Pipeline should pass sample_weight
    X = np.array([[1, 2]])
    X = Xyfn(X)
    pipe = Pipeline([("transf", Transf()), ("clf", FitParamT())])
    pipe.fit(X, y=None)
    assert pipe.score(X) == 3
    assert pipe.score(X, y=None) == 3
    assert pipe.score(X, y=None, sample_weight=None) == 3
    assert pipe.score(X, sample_weight=np.array([2, 3])) == 8


@pytest.mark.parametrize("Pipeline,Xyfn", [
    (_Pipeline, lambda X: X),
    (partial(Pipeline, target_col=None), lambda X: X),
])
def test_pipeline_sample_weight_unsupported(Pipeline, Xyfn):
    # When sample_weight is None it shouldn't be passed
    X = np.array([[1, 2]])
    X = Xyfn(X)
    pipe = Pipeline([("transf", Transf()), ("clf", Mult())])
    pipe.fit(X, y=None)
    assert pipe.score(X) == 3
    assert pipe.score(X, sample_weight=None) == 3

    msg = re.escape(
        "score() got an unexpected keyword argument 'sample_weight'")
    with pytest.raises(TypeError, match=msg):
        pipe.score(X, sample_weight=np.array([2, 3]))


@pytest.mark.parametrize("Pipeline", [_Pipeline, Pipeline])
def test_pipeline_raise_set_params_error(Pipeline):
    # Test pipeline raises set params error message for nested models.
    pipe = Pipeline([("cls", LinearRegression())])

    # expected error message
    error_msg = re.escape(f"Invalid parameter fake for estimator {pipe}. "
                          "Check the list of available parameters "
                          "with `estimator.get_params().keys()`.")

    with pytest.raises(ValueError, match=error_msg):
        pipe.set_params(fake="nope")

    # nested model check
    with pytest.raises(ValueError, match=error_msg):
        pipe.set_params(fake__estimator="nope")


@pytest.mark.parametrize(
    "Pipeline,Xyfn",
    [(_Pipeline, lambda X, y: X),
     (partial(Pipeline, target_col=None), lambda X, y: XyData(X, y))])
def test_pipeline_methods_pca_svm(Pipeline, Xyfn):
    # Test the various methods of the pipeline (pca + svm).
    X = iris.data
    y = iris.target
    X = Xyfn(X, y)
    # Test with PCA + SVC
    clf = SVC(probability=True, random_state=0)
    pca = PCA(svd_solver="full", n_components="mle", whiten=True)
    pipe = Pipeline([("pca", pca), ("svc", clf)])
    pipe.fit(X, y)
    pipe.predict(X)
    pipe.predict_proba(X)
    pipe.predict_log_proba(X)
    pipe.score(X, y)


@pytest.mark.parametrize("Pipeline,Xyfn", [
    (_Pipeline, lambda X: X),
    (partial(Pipeline, target_col=None), lambda X: X),
])
def test_pipeline_score_samples_pca_lof(Pipeline, Xyfn):
    X = iris.data
    X = Xyfn(X)
    # Test that the score_samples method is implemented on a pipeline.
    # Test that the score_samples method on pipeline yields same results as
    # applying transform and score_samples steps separately.
    pca = PCA(svd_solver="full", n_components="mle", whiten=True)
    lof = LocalOutlierFactor(novelty=True)
    pipe = Pipeline([("pca", pca), ("lof", lof)])
    pipe.fit(X)
    # Check the shapes
    assert pipe.score_samples(X).shape == (X.shape[0], )
    # Check the values
    lof.fit(pca.fit_transform(X))
    assert_allclose(pipe.score_samples(X), lof.score_samples(pca.transform(X)))


@pytest.mark.parametrize("make_pipeline,Xyfn", [
    (_make_pipeline, lambda X, y: X),
    (partial(make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
def test_score_samples_on_pipeline_without_score_samples(make_pipeline, Xyfn):
    X = np.array([[1], [2]])
    y = np.array([1, 2])
    X = Xyfn(X, y)
    # Test that a pipeline does not have score_samples method when the final
    # step of the pipeline does not have score_samples defined.
    pipe = make_pipeline(LogisticRegression())
    pipe.fit(X, y)
    with pytest.raises(
            AttributeError,
            match="'LogisticRegression' object has no attribute 'score_samples'",
    ):
        pipe.score_samples(X)


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
def test_pipeline_methods_preprocessing_svm(Pipeline, make_pipeline, Xyfn):
    # Test the various methods of the pipeline (preprocessing + svm).
    X = iris.data
    y = iris.target
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    scaler = StandardScaler()
    pca = PCA(n_components=2, svd_solver="randomized", whiten=True)
    clf = SVC(probability=True, random_state=0, decision_function_shape="ovr")

    X = Xyfn(X, y)

    for preprocessing in [scaler, pca]:
        pipe = Pipeline([("preprocess", preprocessing), ("svc", clf)])
        pipe.fit(X, y)

        # check shapes of various prediction functions
        predict = pipe.predict(X)
        assert predict.shape == (n_samples, )

        proba = pipe.predict_proba(X)
        assert proba.shape == (n_samples, n_classes)

        log_proba = pipe.predict_log_proba(X)
        assert log_proba.shape == (n_samples, n_classes)

        decision_function = pipe.decision_function(X)
        assert decision_function.shape == (n_samples, n_classes)

        pipe.score(X, y)


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
def test_fit_predict_on_pipeline(Pipeline, make_pipeline, Xyfn):
    # test that the fit_predict method is implemented on a pipeline
    # test that the fit_predict on pipeline yields same results as applying
    # transform and clustering steps separately
    scaler = StandardScaler()
    km = KMeans(random_state=0)
    # As pipeline doesn't clone estimators on construction,
    # it must have its own estimators
    scaler_for_pipeline = StandardScaler()
    km_for_pipeline = KMeans(random_state=0)

    # first compute the transform and clustering step separately
    scaled = scaler.fit_transform(iris.data)
    separate_pred = km.fit_predict(scaled)

    # use a pipeline to do the transform and clustering in one step
    pipe = Pipeline([("scaler", scaler_for_pipeline),
                     ("Kmeans", km_for_pipeline)])
    pipeline_pred = pipe.fit_predict(iris.data)

    assert_array_almost_equal(pipeline_pred, separate_pred)


@pytest.mark.parametrize("Pipeline", [
    _Pipeline,
    partial(Pipeline, target_col=None),
])
def test_fit_predict_on_pipeline_without_fit_predict(Pipeline):
    # tests that a pipeline does not have fit_predict method when final
    # step of pipeline does not have fit_predict defined
    scaler = StandardScaler()
    pca = PCA(svd_solver="full")
    pipe = Pipeline([("scaler", scaler), ("pca", pca)])

    msg = "'PCA' object has no attribute 'fit_predict'"
    with pytest.raises(AttributeError, match=msg):
        getattr(pipe, "fit_predict")


@pytest.mark.parametrize("Pipeline", [
    _Pipeline,
    partial(Pipeline, target_col=None),
])
def test_fit_predict_with_intermediate_fit_params(Pipeline):
    # tests that Pipeline passes fit_params to intermediate steps
    # when fit_predict is invoked
    pipe = Pipeline([("transf", TransfFitParams()), ("clf", FitParamT())])
    pipe.fit_predict(X=None, y=None, transf__should_get_this=True,
                     clf__should_succeed=True)
    assert pipe.named_steps["transf"].fit_params["should_get_this"]
    assert pipe.named_steps["clf"].successful
    assert "should_succeed" not in pipe.named_steps["transf"].fit_params


@pytest.mark.parametrize("Pipeline", [
    _Pipeline,
    partial(Pipeline, target_col=None),
])
@pytest.mark.parametrize("method_name",
                         ["predict", "predict_proba", "predict_log_proba"])
def test_predict_methods_with_predict_params(Pipeline, method_name):
    # tests that Pipeline passes predict_* to the final estimator
    # when predict_* is invoked
    pipe = Pipeline([("transf", Transf()), ("clf", DummyEstimatorParams())])
    pipe.fit(None, None)
    method = getattr(pipe, method_name)
    method(X=None, got_attribute=True)

    assert pipe.named_steps["clf"].got_attribute


@pytest.mark.parametrize("Pipeline,Xyfn", [
    (_Pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=None), lambda X, y: np.hstack(
        (X, np.atleast_2d(y).T))),
])
def test_pipeline_transform(Pipeline, Xyfn):
    # Test whether pipeline works with a transformer at the end.
    # Also test pipeline.transform and pipeline.inverse_transform
    X = iris.data
    pca = PCA(n_components=2, svd_solver="full")
    pipeline = Pipeline([("pca", pca)])

    # test transform and fit_transform:
    X_trans = pipeline.fit(X).transform(X)
    X_trans2 = pipeline.fit_transform(X)
    X_trans3 = pca.fit_transform(X)
    assert_array_almost_equal(X_trans, X_trans2)
    assert_array_almost_equal(X_trans, X_trans3)

    X_back = pipeline.inverse_transform(X_trans)
    X_back2 = pca.inverse_transform(X_trans)
    assert_array_almost_equal(X_back, X_back2)


@pytest.mark.parametrize("Pipeline,Xyfn", [
    (_Pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), lambda X, y: np.hstack(
        (X, np.atleast_2d(y).T))),
])
def test_pipeline_fit_transform(Pipeline, Xyfn):
    # Test whether pipeline works with a transformer missing fit_transform
    X = iris.data
    y = iris.target
    X = Xyfn(X, y)
    transf = Transf()
    pipeline = Pipeline([("mock", transf)])

    # test fit_transform:
    X_trans = pipeline.fit_transform(X, y)
    X_trans2 = transf.fit(X, y).transform(X)
    assert_array_almost_equal(X_trans, X_trans2)


@pytest.mark.parametrize("Pipeline", [_Pipeline, Pipeline])
@pytest.mark.parametrize("start, end", [(0, 1), (0, 2), (1, 2), (1, 3),
                                        (None, 1), (1, None), (None, None)])
def test_pipeline_slice(Pipeline, start, end):
    pipe = Pipeline(
        [("transf1", Transf()), ("transf2", Transf()), ("clf", FitParamT())],
        memory="123",
        verbose=True,
    )
    pipe_slice = pipe[start:end]
    # Test class
    assert isinstance(pipe_slice, _Pipeline)
    # Test steps
    assert pipe_slice.steps == pipe.steps[start:end]
    # Test named_steps attribute
    assert (list(pipe_slice.named_steps.items()) == list(
        pipe.named_steps.items())[start:end])
    # Test the rest of the parameters
    pipe_params = pipe.get_params(deep=False)
    pipe_slice_params = pipe_slice.get_params(deep=False)
    del pipe_params["steps"]
    del pipe_slice_params["steps"]
    assert pipe_params == pipe_slice_params
    # Test exception
    msg = "Pipeline slicing only supports a step of 1"
    with pytest.raises(ValueError, match=msg):
        pipe[start:end:-1]


@pytest.mark.parametrize("Pipeline", [_Pipeline, Pipeline])
def test_pipeline_index(Pipeline):
    transf = Transf()
    clf = FitParamT()
    pipe = Pipeline([("transf", transf), ("clf", clf)])
    assert pipe[0] == transf
    assert pipe["transf"] == transf
    assert pipe[-1] == clf
    assert pipe["clf"] == clf

    # should raise an error if slicing out of range
    with pytest.raises(IndexError):
        pipe[3]

    # should raise an error if indexing with wrong element name
    with pytest.raises(KeyError):
        pipe["foobar"]


@pytest.mark.parametrize("Pipeline", [_Pipeline, Pipeline])
def test_set_pipeline_steps(Pipeline):
    transf1 = Transf()
    transf2 = Transf()
    pipeline = Pipeline([("mock", transf1)])
    if Pipeline == _Pipeline:
        assert pipeline.named_steps["mock"] is transf1
    else:
        assert pipeline.named_steps["mock"] == transf1

    # Directly setting attr
    pipeline.steps = [("mock2", transf2)]
    assert "mock" not in pipeline.named_steps
    if Pipeline == _Pipeline:
        assert pipeline.named_steps["mock2"] is transf2
    else:
        assert pipeline.named_steps["mock2"] == transf2
    assert [("mock2", transf2)] == pipeline.steps

    # Using set_params
    pipeline.set_params(steps=[("mock", transf1)])
    assert [("mock", transf1)] == pipeline.steps

    # Using set_params to replace single step
    pipeline.set_params(mock=transf2)
    assert [("mock", transf2)] == pipeline.steps

    # With invalid data
    pipeline.set_params(steps=[("junk", ())])
    msg = re.escape(
        "Last step of Pipeline should implement fit or be the string 'passthrough'."
    )
    with pytest.raises(TypeError, match=msg):
        pipeline.fit([[1]], [1])

    with pytest.raises(TypeError, match=msg):
        pipeline.fit_transform([[1]], [1])


@pytest.mark.parametrize("Pipeline", [_Pipeline, Pipeline])
def test_pipeline_named_steps(Pipeline):
    transf = Transf()
    mult2 = Mult(mult=2)
    pipeline = Pipeline([("mock", transf), ("mult", mult2)])

    # Test access via named_steps bunch object
    assert "mock" in pipeline.named_steps
    assert "mock2" not in pipeline.named_steps
    if Pipeline is _Pipeline:
        assert pipeline.named_steps.mock is transf
        assert pipeline.named_steps.mult is mult2
    else:
        assert pipeline.named_steps.mock == transf
        assert pipeline.named_steps.mult == mult2

    # Test bunch with conflict attribute of dict
    pipeline = Pipeline([("values", transf), ("mult", mult2)])
    if Pipeline is _Pipeline:
        assert pipeline.named_steps.values is not transf
        assert pipeline.named_steps.mult is mult2
    else:
        assert pipeline.named_steps.values != transf
        assert pipeline.named_steps.mult == mult2


@pytest.mark.parametrize("Pipeline,Xyfn", [
    (_Pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), lambda X, y: np.hstack(
        (X, np.atleast_2d(y).T))),
])
@pytest.mark.parametrize("passthrough", [None, "passthrough"])
def test_pipeline_correctly_adjusts_steps(Pipeline, Xyfn, passthrough):
    X = np.array([[1]])
    y = np.array([1])
    X = Xyfn(X, y)
    mult2 = Mult(mult=2)
    mult3 = Mult(mult=3)
    mult5 = Mult(mult=5)

    pipeline = Pipeline([("m2", mult2), ("bad", passthrough), ("m3", mult3),
                         ("m5", mult5)])

    if Pipeline is _Pipeline:
        pipeline.fit(X, y)
    else:
        pipeline.fit(X)
    expected_names = ["m2", "bad", "m3", "m5"]
    actual_names = [name for name, _ in pipeline.steps]
    assert expected_names == actual_names


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
@pytest.mark.parametrize("passthrough", [None, "passthrough"])
def test_set_pipeline_step_passthrough(Pipeline, make_pipeline, Xyfn,
                                       passthrough):
    X = np.array([[1]])
    y = np.array([1])
    X = Xyfn(X, y)
    mult2 = Mult(mult=2)
    mult3 = Mult(mult=3)
    mult5 = Mult(mult=5)

    def make():
        return Pipeline([("m2", mult2), ("m3", mult3), ("last", mult5)])

    pipeline = make()

    exp = 2 * 3 * 5
    if Pipeline is _Pipeline:
        assert_array_equal([[exp]], pipeline.fit_transform(X, y))
        assert_array_equal([exp], pipeline.fit(X).predict(X))
        assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    else:
        assert_array_equal([[exp]], pipeline.fit_transform(X, y).X)
        assert_array_equal([exp], pipeline.fit(X).predict(X))
        assert_array_equal(X.X, pipeline.inverse_transform([[exp]]))

    pipeline.set_params(m3=passthrough)
    exp = 2 * 5
    if Pipeline is _Pipeline:
        assert_array_equal([[exp]], pipeline.fit_transform(X, y))
        assert_array_equal([exp], pipeline.fit(X).predict(X))
        assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    else:
        assert_array_equal([[exp]], pipeline.fit_transform(X, y).X)
        assert_array_equal([exp], pipeline.fit(X).predict(X))
        assert_array_equal(X.X, pipeline.inverse_transform([[exp]]))
    expected = {
        "steps": pipeline.steps,
        "m2": mult2,
        "m3": passthrough,
        "last": mult5,
        "memory": None,
        "m2__mult": 2,
        "last__mult": 5,
        "verbose": False,
    }
    if Pipeline is not _Pipeline:
        expected.update(dict(ofmt=None, target_col=-1))
    assert pipeline.get_params(deep=True) == expected

    pipeline.set_params(m2=passthrough)
    exp = 5
    if Pipeline is _Pipeline:
        assert_array_equal([[exp]], pipeline.fit_transform(X, y))
        assert_array_equal([exp], pipeline.fit(X).predict(X))
        assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    else:
        assert_array_equal([[exp]], pipeline.fit_transform(X, y).X)
        assert_array_equal([exp], pipeline.fit(X).predict(X))
        assert_array_equal(X.X, pipeline.inverse_transform([[exp]]))

    # for other methods, ensure no AttributeErrors on None:
    other_methods = [
        "predict_proba",
        "predict_log_proba",
        "decision_function",
        "transform",
        "score",
    ]
    for method in other_methods:
        getattr(pipeline, method)(X)

    pipeline.set_params(m2=mult2)
    exp = 2 * 5
    if Pipeline is _Pipeline:
        assert_array_equal([[exp]], pipeline.fit_transform(X, y))
        assert_array_equal([exp], pipeline.fit(X).predict(X))
        assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    else:
        assert_array_equal([[exp]], pipeline.fit_transform(X, y).X)
        assert_array_equal([exp], pipeline.fit(X).predict(X))
        assert_array_equal(X.X, pipeline.inverse_transform([[exp]]))

    pipeline = make()
    pipeline.set_params(last=passthrough)
    # mult2 and mult3 are active
    exp = 6
    if Pipeline is _Pipeline:
        assert_array_equal([[exp]], pipeline.fit(X, y).transform(X))
        assert_array_equal([[exp]], pipeline.fit_transform(X, y))
        assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    else:
        assert_array_equal([[exp]], pipeline.fit(X, y).transform(X).X)
        assert_array_equal([[exp]], pipeline.fit_transform(X, y).X)
        assert_array_equal(X.X, pipeline.inverse_transform([[exp]]))

    msg = "'str' object has no attribute 'predict'"
    with pytest.raises(AttributeError, match=msg):
        getattr(pipeline, "predict")

    # Check 'passthrough' step at construction time
    exp = 2 * 5
    pipeline = Pipeline([("m2", mult2), ("m3", passthrough), ("last", mult5)])
    if Pipeline is _Pipeline:
        assert_array_equal([[exp]], pipeline.fit_transform(X, y))
        assert_array_equal([exp], pipeline.fit(X).predict(X))
        assert_array_equal(X, pipeline.inverse_transform([[exp]]))
    else:
        assert_array_equal([[exp]], pipeline.fit_transform(X, y).X)
        assert_array_equal([exp], pipeline.fit(X).predict(X))
        assert_array_equal(X.X, pipeline.inverse_transform([[exp]]))


@pytest.mark.parametrize("make_pipeline", [_make_pipeline, make_pipeline])
def test_pipeline_ducktyping(make_pipeline):
    pipeline = make_pipeline(Mult(5))
    pipeline.predict
    pipeline.transform
    pipeline.inverse_transform

    pipeline = make_pipeline(Transf())
    assert not hasattr(pipeline, "predict")
    pipeline.transform
    pipeline.inverse_transform

    pipeline = make_pipeline("passthrough")
    assert pipeline.steps[0] == ("passthrough", "passthrough")
    assert not hasattr(pipeline, "predict")
    pipeline.transform
    pipeline.inverse_transform

    pipeline = make_pipeline(Transf(), NoInvTransf())
    assert not hasattr(pipeline, "predict")
    pipeline.transform
    assert not hasattr(pipeline, "inverse_transform")

    pipeline = make_pipeline(NoInvTransf(), Transf())
    assert not hasattr(pipeline, "predict")
    pipeline.transform
    assert not hasattr(pipeline, "inverse_transform")


@pytest.mark.parametrize("make_pipeline", [_make_pipeline, make_pipeline])
def test_make_pipeline(make_pipeline):
    t1 = Transf()
    t2 = Transf()
    pipe = make_pipeline(t1, t2)
    assert isinstance(pipe, _Pipeline)
    assert pipe.steps[0][0] == "transf-1"
    assert pipe.steps[1][0] == "transf-2"

    pipe = make_pipeline(t1, t2, FitParamT())
    assert isinstance(pipe, _Pipeline)
    assert pipe.steps[0][0] == "transf-1"
    assert pipe.steps[1][0] == "transf-2"
    assert pipe.steps[2][0] == "fitparamt"


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
def test_classes_property(Pipeline, make_pipeline, Xyfn):
    X = iris.data
    y = iris.target
    X = Xyfn(X, y)

    reg = make_pipeline(SelectKBest(k=1), LinearRegression())
    reg.fit(X, y)
    with pytest.raises(AttributeError):
        getattr(reg, "classes_")

    clf = make_pipeline(SelectKBest(k=1), LogisticRegression(random_state=0))
    with pytest.raises(AttributeError):
        getattr(clf, "classes_")
    clf.fit(X, y)
    assert_array_equal(clf.classes_, np.unique(y))


@pytest.mark.parametrize("Pipeline", [_Pipeline, Pipeline])
def test_set_params_nested_pipeline(Pipeline):
    estimator = Pipeline([("a", Pipeline([("b", DummyRegressor())]))])
    estimator.set_params(a__b__alpha=0.001, a__b=Lasso())
    estimator.set_params(a__steps=[("b", LogisticRegression())], a__b__C=5)


@pytest.mark.parametrize("Pipeline,Xyfn", [
    (_Pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), lambda X, y: np.hstack(
        (X, np.atleast_2d(y).T))),
])
def test_pipeline_wrong_memory(Pipeline, Xyfn):
    # Test that an error is raised when memory is not a string or a Memory
    # instance
    X = iris.data
    y = iris.target
    X = Xyfn(X, y)
    # Define memory as an integer
    memory = 1
    cached_pipe = Pipeline([("transf", DummyTransf()), ("svc", SVC())],
                           memory=memory)

    msg = re.escape(
        "'memory' should be None, a string or have the same interface "
        "as joblib.Memory. Got memory='1' instead.")
    with pytest.raises(ValueError, match=msg):
        cached_pipe.fit(X, y)


class DummyMemory:

    def cache(self, func):
        return func


class WrongDummyMemory:
    pass


@pytest.mark.parametrize("Pipeline", [_Pipeline, Pipeline])
def test_pipeline_with_cache_attribute(Pipeline):
    X = np.array([[1, 2]])
    pipe = Pipeline([("transf", Transf()), ("clf", Mult())],
                    memory=DummyMemory())
    pipe.fit(X, y=None)
    dummy = WrongDummyMemory()
    pipe = Pipeline([("transf", Transf()), ("clf", Mult())], memory=dummy)
    msg = re.escape(
        "'memory' should be None, a string or have the same interface "
        f"as joblib.Memory. Got memory='{dummy}' instead.")
    with pytest.raises(ValueError, match=msg):
        pipe.fit(X)


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
def test_pipeline_memory(Pipeline, make_pipeline, Xyfn):
    X = iris.data
    y = iris.target
    X = Xyfn(X, y)
    cachedir = mkdtemp()
    try:
        if parse_version(joblib.__version__) < parse_version("0.12"):
            # Deal with change of API in joblib
            memory = joblib.Memory(cachedir=cachedir, verbose=10)
        else:
            memory = joblib.Memory(location=cachedir, verbose=10)
        # Test with Transformer + SVC
        clf = SVC(probability=True, random_state=0)
        transf = DummyTransf()
        pipe = Pipeline([("transf", clone(transf)), ("svc", clf)])
        cached_pipe = Pipeline([("transf", transf), ("svc", clf)],
                               memory=memory)

        # Memoize the transformer at the first fit
        cached_pipe.fit(X, y)
        pipe.fit(X, y)
        # Get the time stamp of the transformer in the cached pipeline
        ts = cached_pipe.named_steps["transf"].timestamp_
        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(pipe.predict(X), cached_pipe.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe.predict_proba(X))
        assert_array_equal(pipe.predict_log_proba(X),
                           cached_pipe.predict_log_proba(X))
        assert_array_equal(pipe.score(X, y), cached_pipe.score(X, y))
        assert_array_equal(pipe.named_steps["transf"].means_,
                           cached_pipe.named_steps["transf"].means_)
        assert not hasattr(transf, "means_")
        # Check that we are reading the cache while fitting
        # a second time
        cached_pipe.fit(X, y)
        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(pipe.predict(X), cached_pipe.predict(X))
        assert_array_equal(pipe.predict_proba(X), cached_pipe.predict_proba(X))
        assert_array_equal(pipe.predict_log_proba(X),
                           cached_pipe.predict_log_proba(X))
        assert_array_equal(pipe.score(X, y), cached_pipe.score(X, y))
        assert_array_equal(pipe.named_steps["transf"].means_,
                           cached_pipe.named_steps["transf"].means_)
        assert ts == cached_pipe.named_steps["transf"].timestamp_
        # Create a new pipeline with cloned estimators
        # Check that even changing the name step does not affect the cache hit
        clf_2 = SVC(probability=True, random_state=0)
        transf_2 = DummyTransf()
        cached_pipe_2 = Pipeline([("transf_2", transf_2), ("svc", clf_2)],
                                 memory=memory)
        cached_pipe_2.fit(X, y)

        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(pipe.predict(X), cached_pipe_2.predict(X))
        assert_array_equal(pipe.predict_proba(X),
                           cached_pipe_2.predict_proba(X))
        assert_array_equal(pipe.predict_log_proba(X),
                           cached_pipe_2.predict_log_proba(X))
        assert_array_equal(pipe.score(X, y), cached_pipe_2.score(X, y))
        assert_array_equal(
            pipe.named_steps["transf"].means_,
            cached_pipe_2.named_steps["transf_2"].means_,
        )
        assert ts == cached_pipe_2.named_steps["transf_2"].timestamp_
    finally:
        shutil.rmtree(cachedir)


@pytest.mark.parametrize("make_pipeline,Xyfn", [
    (_make_pipeline, lambda X, y: X),
    (partial(make_pipeline, target_col=-1), lambda X, y: np.hstack(
        (X, np.atleast_2d(y).T))),
])
def test_make_pipeline_memory(make_pipeline, Xyfn):
    cachedir = mkdtemp()
    if parse_version(joblib.__version__) < parse_version("0.12"):
        # Deal with change of API in joblib
        memory = joblib.Memory(cachedir=cachedir, verbose=10)
    else:
        memory = joblib.Memory(location=cachedir, verbose=10)
    pipeline = make_pipeline(DummyTransf(), SVC(), memory=memory)
    assert pipeline.memory is memory
    pipeline = make_pipeline(DummyTransf(), SVC())
    assert pipeline.memory is None
    assert len(pipeline) == 2

    shutil.rmtree(cachedir)


class FeatureNameSaver(BaseEstimator):

    def fit(self, X, y=None):
        self._check_feature_names(X, reset=True)
        return self

    def transform(self, X, y=None):
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
def test_features_names_passthrough(Pipeline, make_pipeline, Xyfn):
    """Check pipeline.get_feature_names_out with passthrough"""
    pipe = Pipeline(steps=[
        ("names", FeatureNameSaver()),
        ("pass", "passthrough"),
        ("clf", LogisticRegression()),
    ])
    iris = load_iris()

    X = iris.data
    y = iris.target
    X = Xyfn(X, y)

    pipe.fit(X, y)
    assert_array_equal(pipe[:-1].get_feature_names_out(iris.feature_names),
                       iris.feature_names)


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
def test_feature_names_count_vectorizer(Pipeline, make_pipeline, Xyfn):
    """Check pipeline.get_feature_names_out with vectorizers"""
    pipe = Pipeline(steps=[("vect",
                            CountVectorizer()), ("clf", LogisticRegression())])
    X = JUNK_FOOD_DOCS
    y = ["pizza" in x for x in JUNK_FOOD_DOCS]
    X = Xyfn(X, y)
    pipe.fit(X, y)
    assert_array_equal(
        pipe[:-1].get_feature_names_out(),
        ["beer", "burger", "coke", "copyright", "pizza", "the"],
    )
    assert_array_equal(
        pipe[:-1].get_feature_names_out("nonsense_is_ignored"),
        ["beer", "burger", "coke", "copyright", "pizza", "the"],
    )


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
def test_pipeline_feature_names_out_error_without_definition(
        Pipeline, make_pipeline, Xyfn):
    """Check that error is raised when a transformer does not define
    `get_feature_names_out`."""
    pipe = Pipeline(steps=[("notrans", NoTrans())])
    iris = load_iris()
    X = iris.data
    y = iris.target
    X = Xyfn(X, y)
    pipe.fit(X, y)

    msg = "does not provide get_feature_names_out"
    with pytest.raises(AttributeError, match=msg):
        pipe.get_feature_names_out()


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
def test_pipeline_param_error(Pipeline, make_pipeline, Xyfn):
    clf = make_pipeline(LogisticRegression())
    with pytest.raises(
            ValueError,
            match="Pipeline.fit does not accept the sample_weight parameter"):
        X = [[0], [0]]
        y = [0, 1]
        X = Xyfn(X, y)
        clf.fit(X, y, sample_weight=[1, 1])


parameter_grid_test_verbose = ((est, pattern, method) for (
    est, pattern), method in itertools.product(
        [
            (
                Pipeline([("transf", Transf()), ("clf", FitParamT())]),
                r"\[Pipeline\].*\(step 1 of 2\) Processing transf.* total=.*\n"
                r"\[Pipeline\].*\(step 2 of 2\) Processing clf.* total=.*\n$",
            ),
        ],
        ["fit", "fit_transform", "fit_predict"],
    ) if hasattr(est, method) and not (method == "fit_transform" and hasattr(
        est, "steps") and isinstance(est.steps[-1][1], FitParamT)))


@pytest.mark.parametrize("est, pattern, method", parameter_grid_test_verbose)
def test_verbose(est, method, pattern, capsys):
    func = getattr(est, method)

    X = [[1, 2, 3], [4, 5, 6]]
    y = [[7], [8]]

    if est.__class__ is not _Pipeline:
        X = XyData(X, y)

    est.set_params(verbose=False)
    func(X, y)
    assert not capsys.readouterr().out, "Got output for verbose=False"

    est.set_params(verbose=True)
    func(X, y)
    assert re.match(pattern, capsys.readouterr().out)


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
def test_n_features_in_pipeline(Pipeline, make_pipeline, Xyfn):
    # make sure pipelines delegate n_features_in to the first step

    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]
    X = Xyfn(X, y)

    ss = StandardScaler()
    gbdt = HistGradientBoostingClassifier()
    pipe = make_pipeline(ss, gbdt)
    assert not hasattr(pipe, "n_features_in_")
    pipe.fit(X, y)
    assert pipe.n_features_in_ == ss.n_features_in_ == 2

    # if the first step has the n_features_in attribute then the pipeline also
    # has it, even though it isn't fitted.
    ss = StandardScaler()
    gbdt = HistGradientBoostingClassifier()
    pipe = make_pipeline(ss, gbdt)
    if Pipeline is _Pipeline:
        ss.fit(X, y)
    else:
        ss.fit(X.X, y)
    assert pipe.n_features_in_ == ss.n_features_in_ == 2
    assert not hasattr(gbdt, "n_features_in_")


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
def test_pipeline_missing_values_leniency(Pipeline, make_pipeline, Xyfn):
    # check that pipeline let the missing values validation to
    # the underlying transformers and predictors.
    X, y = iris.data, iris.target
    mask = np.random.choice([1, 0], X.shape, p=[0.1, 0.9]).astype(bool)
    X[mask] = np.nan
    X = Xyfn(X, y)
    pipe = make_pipeline(SimpleImputer(), LogisticRegression())
    assert pipe.fit(X, y).score(X, y) > 0.4


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
@pytest.mark.parametrize("passthrough", [None, "passthrough"])
def test_pipeline_get_tags_none(Pipeline, make_pipeline, Xyfn, passthrough):
    # Checks that tags are set correctly when the first transformer is None or
    # 'passthrough'
    # Non-regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/18815
    pipe = make_pipeline(passthrough, SVC())
    assert not pipe._get_tags()["pairwise"]


# # FIXME: Replace this test with a full `check_estimator` once we have API only
# # checks.
@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
@pytest.mark.parametrize("Predictor", [MinimalRegressor, MinimalClassifier])
def test_search_cv_using_minimal_compatible_estimator(Pipeline, make_pipeline,
                                                      Xyfn, Predictor):
    # Check that third-party library estimators can be part of a pipeline
    # and tuned by grid-search without inheriting from BaseEstimator.
    rng = np.random.RandomState(0)
    X, y = rng.randn(25, 2), np.array([0] * 5 + [1] * 20)
    X = Xyfn(X, y)

    model = Pipeline([("transformer", MinimalTransformer()),
                      ("predictor", Predictor())])
    model.fit(X, y)

    y_pred = model.predict(X)
    if is_classifier(model):
        assert_array_equal(y_pred, 1)
        assert model.score(X, y) == pytest.approx(accuracy_score(y, y_pred))
    else:
        assert_allclose(y_pred, y.mean())
        assert model.score(X, y) == pytest.approx(r2_score(y, y_pred))


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
def test_pipeline_check_if_fitted(Pipeline, make_pipeline, Xyfn):

    class Estimator(BaseEstimator):

        def fit(self, X, y):
            self.fitted_ = True
            return self

    pipeline = Pipeline([("clf", Estimator())])
    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline)
    X = iris.data
    y = iris.target
    X = Xyfn(X, y)
    pipeline.fit(X, y)
    check_is_fitted(pipeline)


@pytest.mark.parametrize("Pipeline,make_pipeline,Xyfn", [
    (_Pipeline, _make_pipeline, lambda X, y: X),
    (partial(Pipeline, target_col=-1), partial(
        make_pipeline, target_col=-1), lambda X, y: XyData(X, y)),
])
def test_pipeline_get_feature_names_out_passes_names_through(
        Pipeline, make_pipeline, Xyfn):
    """Check that pipeline passes names through.

    Non-regresion test for #21349.
    """
    X, y = iris.data, iris.target
    X = Xyfn(X, y)

    class AddPrefixStandardScalar(StandardScaler):

        def get_feature_names_out(self, input_features=None):
            names = super().get_feature_names_out(input_features=input_features)
            return np.asarray([f"my_prefix_{name}" for name in names],
                              dtype=object)

    pipe = make_pipeline(AddPrefixStandardScalar(), StandardScaler())
    pipe.fit(X, y)

    input_names = iris.feature_names
    feature_names_out = pipe.get_feature_names_out(input_names)

    assert_array_equal(feature_names_out,
                       [f"my_prefix_{name}" for name in input_names])
