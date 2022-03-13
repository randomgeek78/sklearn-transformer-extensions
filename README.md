# sklearn-transformer-extensions

This python package attempts to make building end-to-end pipelines using
`scikit-learn` easier and more intuitive. It provides a collection of new
classes and drop-in replacements to some existing `scikit-learn` classes that
work together to make it very easy to build very sophisticated pipelines with
ease. Besides, there are no functional limitations to using this new workflow.

The pain points that the package addresses are:
- `scikit-learn` API keeps the features and labels separate. This makes it
  impossible to remove outliers from the training data as part of an integrated
  pipeline when using the standard `scikit-learn` API.
- Cumbersome to build decentralized, multi-step pipelines. See note and example.

## Issue 1: `scikit-learn` API keeps the features and labels separate

The first issue is highlighted in the following code snippet. We want to take
an input, filter it for outliers, then fit a model on the filtered train set.
The filtering step cannot be done as part of a pipeline using the current
`scikit-learn` API.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline

df_train = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [0, 2, 4, 6, 100]})
print(df_train)
#    x    y
# 0  0    0
# 1  1    2
# 2  2    4
# 3  3    6
# 4  4  100   # outlier

# Remove outlier
lof = LocalOutlierFactor(n_neighbors=2)
mask = lof.fit_predict(df_train)
print(mask)
# [ 1  1  1  1 -1]
df_train_filtered = df_train.iloc[mask > 0, :]

# Fit model to cleaned data
lr = LinearRegression()
lr.fit(df_train_filtered[["x"]], df_train_filtered["y"])

# Predict
df_test = pd.DataFrame({"x": np.arange(10, 17)})
print(lr.predict(df_test[["x"]]))
# [20. 22. 24. 26. 28. 30. 32.]
```

Using the current `scikit-learn` API, it is not possible to create a pipeline
with both the outlier detector and the linear regressor. The solution is to
keep the X and y together in the external API and internally deal with
interfacing with `scikit-learn` transformers/estimators. We implemented the
`XyAdapter` that knows how to split the joint Xy data-structure and interface
with `scikit-learn` APIs.

```python
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn_transformer_extensions import XyAdapter

df_train = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [0, 2, 4, 6, 100]})
print(df_train)
#    x    y
# 0  0    0
# 1  1    2
# 2  2    4
# 3  3    6
# 4  4  100   # outlier

class LocalOutlierFilter(TransformerMixin, BaseEstimator):
  def __init__(self, **kwargs):
    self.transformer = LocalOutlierFactor(**kwargs)
  # We want to filter the train data
  def fit_transform(self, X, y=None, **fit_params):
    self.transformer_ = clone(self.transformer)
    mask = self.transformer_.fit_predict(X, y, **fit_params)
    if hasattr(X, "iloc"):
      X = X.iloc
    return X[mask > 0, :]
  # We don't filter the test data
  def transform(self, X):
    return X

lof = LocalOutlierFilter(n_neighbors=2)
lr = XyAdapter(LinearRegression(), "y")

# Single pipeline w/ train set filtering possible
# This pipeline can also be used to tune filter parameters like n_neighbors
# using grid search
p = make_pipeline(lof, lr)

# Train
p.fit(df_train)

# Predict
df_test = pd.DataFrame({"x": np.arange(10, 17)})
print(p.predict(df_test[["x"]]))
# [20. 22. 24. 26. 28. 30. 32.]

# Check if the filter is actually filtering train set
p[0].fit_transform(df_train)
#    x  y
# 0  0  0
# 1  1  2
# 2  2  4
# 3  3  6
```

## Issue 2: 

