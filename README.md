# sklearn-transformer-extensions

This python package attempts to make building end-to-end pipelines using
scikit-learn and pandas DataFrames easier and more intuitive. It provides a
collection of new classes and drop-in replacements of some existing
scikit-learn classes that work together to make it very easy to build very
sophisticated scikit-learn pipelines with ease. Besides, there are no
functional limitations to using this new workflow - anything that can be
achieved with the original scikit-learn API like cross-validation, grid-search,
etc - is also achievable with this new work. 

## Example workflow

In this example, we create a end-to-end example using an example from [Practical Machine Learning with Python](https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch01_Machine_Learning_Basics/Predicting%20Student%20Recommendation%20Machine%20Learning%20Pipeline.ipynb)

```python
>>> import pandas as pd
>>> from io import StringIO
>>> from sklearn.compose import make_column_transformer
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.preprocessing import OneHotEncoder
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.pipeline import make_pipeline

>>> from sklearn_transformer_extensions import XyAdapter

>>> train: pd.DataFrame = pd.read_csv(  # type: ignore
...     StringIO("""
... Name,OverallGrade,Obedient,ResearchScore,ProjectScore,Recommend
... Henry,A,Y,90,85,Yes
... John,C,N,85,51,Yes
... David,F,N,10,17,No
... Holmes,B,Y,75,71,No
... Marvin,E,N,20,30,No
... Simon,A,Y,92,79,Yes
... Robert,B,Y,60,59,No
... Trent,C,Y,75,33,No
... """))

```

The train data structure contains both features and labels. They are kept
together in our workflow.

```
>>> train.head(2)
    Name OverallGrade Obedient  ResearchScore  ProjectScore Recommend
0  Henry            A        Y             90            85       Yes
1   John            C        N             85            51       Yes

```

The XyAdapter internally splits X and y and forwards them to whatever is the
underlying transformer instance. It then joins the the transformed X features
and the original y labels and returns the combined Xy. This way, externally,
the Xy are always kept together.

```python
>>> ct = XyAdapter(
...     make_column_transformer(
...         (StandardScaler(), ["ResearchScore", "ProjectScore"]),
...         (OneHotEncoder(handle_unknown='ignore'), ['OverallGrade', 'Obedient']),
...         verbose_feature_names_out=False,
...     ), target_col='Recommend')

>>> train_ = ct.fit_transform(train)
>>> train_.head(2)
   ResearchScore  ProjectScore  ...  Obedient_Y  Recommend
0       0.899583      1.376650  ...         1.0        Yes
1       0.730648     -0.091777  ...         0.0        Yes
<BLANKLINE>
[2 rows x 10 columns]

>>> train.shape, train_.shape
((8, 6), (8, 10))

>>> print(all(train['Recommend'] == train_['Recommend']))
True

```

XyAdapter is also used on estimators just like with transformers

```python
>>> lr = XyAdapter(LogisticRegression(), target_col='Recommend')

```

The transformed features can be directly used to fit the estimator.

```python
>>> lr.fit(train_)
LogisticRegression()
>>> lr.predict_proba(train_)[:, 1].sum()
3.000028844251478

```

Or we can create a pipeline with adapted instances

```python
>>> p = make_pipeline(ct, lr)

```

The pipeline is fitted as usual. Note that no labels are provided, i.e. y=None.
The input is expected to contain both X and y.

```python
>>> p = p.fit(train)
>>> p[:-1].transform(train).head(2)
   ResearchScore  ProjectScore  ...  Obedient_Y  Recommend
0       0.899583      1.376650  ...         1.0        Yes
1       0.730648     -0.091777  ...         0.0        Yes
<BLANKLINE>
[2 rows x 10 columns]

>>> p.predict_proba(train)[:, 1].sum()
3.000028844251478

```
