# sklearn-transformer-extensions

This python package attempts to make building pipelines using `scikit-learn`
easier and more intuitive. For some use-cases, keeping the features (X) and the
labels (y) together in the same data-structure provides more flexibility when
building end-to-end pipelines. This package provides the `XyAdapter` class
factory that can wrap around any scikit-learn transformer/estimator. The
wrapped class retains the original class' API and methods while providing an
additional interface to call the methods - it allows taking a `XyData` type
object as the X argument in which case this object is used to provide both the
features and labels.

In addition, this package provides drop-in replacements to `ColumnTransformer`
and `FunctionTransformer` that extend their functionality and make them more
easier to use with pandas data-structures. 

To motivate why we would like to keep features and labels together, we consider
the use-case where we would like to filter outliers from input before fitting a
model to it. The filtering step cannot be done as part of a pipeline using the
current `scikit-learn` API. Our goal is to overcome this limitation by using
`XyAdapter` to build a single pipeline that does both filtering and modeling.
The unified pipeline can also be used for for end-to-end grid search and
cross-validation.

```python
>>> import pandas as pd
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.neighbors import LocalOutlierFactor
>>> from sklearn.pipeline import make_pipeline

# Create data with outlier (X, y) = (4, 100)
>>> df_train = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [0, 2, 4, 6, 100]})
>>> print(df_train)
   x    y
0  0    0
1  1    2
2  2    4
3  3    6
4  4  100

# Remove outlier
>>> lof = LocalOutlierFactor(n_neighbors=2)
>>> mask = lof.fit_predict(df_train)
>>> print(mask)
[ 1  1  1  1 -1]

# Filter outlier
>>> df_filtered = df_train.iloc[mask > 0, :]

# Fit model to cleaned data
>>> lr = LinearRegression()
>>> lr.fit(df_filtered[["x"]], df_filtered["y"])
LinearRegression()

# Predict
>>> df_test = pd.DataFrame({"x": np.arange(10, 17)})
>>> print(lr.predict(df_test[["x"]]))
[20. 22. 24. 26. 28. 30. 32.]

```

The filtering step cannot be combined with the modeling step as part of a
unified pipeline. This is because, at any step of the pipeline, a transformer
only gets to influence the features and not the labels that are seen by the
steps that follow. Thus, transformer steps that affect the number of rows will
make the features and labels mismatched.

The solution is to keep the X and y together in the external API of any
individual pipeline step so both the features and labels are passed around
together as they move through the pipeline and are thus always in sync. We
implemented the `XyAdapter` that receives the combined X and y object and knows
how to split them into the features and labels and transparently interface with
a `scikit-learn` transformer or estimator method that require them to be
separate function arguments. Upon receiving the transformed features from the
underlying transformer, the `XyAdapter` class then packages the transformed
features and original labels and returns the combined data-structure. This way,
external to the `XyAdapter` class, a combined data-structure that contains both
the transformed features and labels is passed around as the "features" while
the labels as far as the pipeline is concerned is always None. As far as the
pipeline is concerned, it is dealing with an unsupervised learning scenario.

```python
>>> import pandas as pd
>>> import numpy as np
>>> from sklearn.base import BaseEstimator, TransformerMixin, clone
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.neighbors import LocalOutlierFactor
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.model_selection import GridSearchCV
>>> from sklearn.model_selection import KFold
>>> from sklearn_transformer_extensions import XyAdapter, XyData

>>> df_train = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [0, 2, 4, 6, 100]})
>>> print(df_train)
   x    y
0  0    0
1  1    2
2  2    4
3  3    6
4  4  100


>>> class LocalOutlierFilter(LocalOutlierFactor):
...   # We want to filter the train data
...   def fit_transform(self, Xy, _=None, **fit_params):
...     assert type(Xy) == XyData
...     X, y = Xy
...     y = np.atleast_2d(y).T
...     X = np.hstack((X, y))
...     mask = self.fit_predict(X, **fit_params)
...     X, y = Xy
...     return XyData(X.iloc[mask > 0, :], y.iloc[mask > 0])
...   # We don't filter the test data
...   def transform(self, X):
...     return X

>>> lof = LocalOutlierFilter(n_neighbors=2)
>>> lr = XyAdapter(LinearRegression)()

# Single pipeline w/ train set filtering possible
# This pipeline can also be used to tune filter parameters like n_neighbors
# using grid search
>>> p = make_pipeline(lof, lr)

# Train
>>> X, y = df_train.loc[:, df_train.columns != 'y'], df_train.loc[:, 'y']
>>> Xy = XyData(X, y)
>>> p.fit(Xy)
Pipeline(steps=[('localoutlierfilter', LocalOutlierFilter(n_neighbors=2)),
                ('linearregression', LinearRegression())])

# Predict
>>> df_test = pd.DataFrame({"x": np.arange(10, 17)})
>>> print(p.predict(df_test[["x"]]))
[20. 22. 24. 26. 28. 30. 32.]

# Check if the filter is actually filtering train set
>>> Xty = p[0].fit_transform(Xy)
>>> print(Xty)
XyData(X=pandas(shape=(4, 1)), y=pandas(shape=(4,)))

>>> pd.concat(Xty, axis=1)
   x  y
0  0  0
1  1  2
2  2  4
3  3  6

# Perform joint grid search of both the filtering step and the modeling step.
>>> kfolds = KFold(n_splits=2, shuffle=True, random_state=42)
>>> gs = GridSearchCV(
...   p, param_grid={
...       "localoutlierfilter__n_neighbors": (1, 2),
...       "linearregression__fit_intercept": (True, False),
...   }, refit=False, cv=kfolds, return_train_score=True, error_score='raise')

>>> gs.fit(Xy)
GridSearchCV(cv=KFold(n_splits=2, random_state=42, shuffle=True),
             error_score='raise',
             estimator=Pipeline(steps=[('localoutlierfilter',
                                        LocalOutlierFilter(n_neighbors=2)),
                                       ('linearregression',
                                        LinearRegression())]),
             param_grid={'linearregression__fit_intercept': (True, False),
                         'localoutlierfilter__n_neighbors': (1, 2)},
             refit=False, return_train_score=True)

>>> results = pd.DataFrame({
...   "n_neighbors": gs.cv_results_['param_localoutlierfilter__n_neighbors'],
...   "fit_intercept": gs.cv_results_['param_linearregression__fit_intercept'],
...   "mean_train": gs.cv_results_['mean_train_score'],
...   "std_train": gs.cv_results_['std_train_score'],
...   "mean_test": gs.cv_results_['mean_test_score'],
...   "std_test": gs.cv_results_['std_test_score'],
... })
>>> print(results)
  n_neighbors fit_intercept  mean_train  std_train   mean_test    std_test
0           1          True    0.325542   0.674458    0.325542    0.674458
1           2          True    0.951824   0.048176 -135.223211  134.874295
2           1         False    0.325542   0.674458    0.325542    0.674458
3           2         False    0.839415   0.160585  -76.445433   76.096517

```

Since both the filter and the estimator are part of the same pipeline, we were
able to jointly optimize the parameters for both. 

Apart from `XyAdapter`, the drop-in replacements to `ColumnTransformer` and
`FunctionTransformer` provide additional functionality. Please refer to their
documentation for more details.
