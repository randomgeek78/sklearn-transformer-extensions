# sklearn-transformer-extensions

This python package attempts to make building end-to-end pipelines using
`scikit-learn` easier and more intuitive. It provides a collection of new
classes and drop-in replacements to some existing `scikit-learn` classes that
work together to make it very easy to build very sophisticated pipelines with
ease. Besides, there are no functional limitations to using this new workflow.

The pain points that the package addresses are:
- `scikit-learn` API keeps the features and labels separate. This makes it
  impossible to remove outliers from the training data as part of an integrated
  pipeline using the standard `scikit-learn` API.
- Cumbersome to create new features while also retaining old features with
  their original feature names, i.e. not prefixing them to avoid name clashes.
  See note.

In practice, these pain points make it very difficult to write complex
end-to-end pipelines that do everything from filtering to feature engineering
to model tuning using cross-validation while also being easy and intuitive to
build.

This package explores the following ideas:
- Keep the features (X) and labels (y) together and pass them around together.
  The combined (X, y) data structure is received by an adapter (`XyAdapter`) that
  interfaces with an underlying transformer or estimator instance. All method
  calls to the adapter are transparently forwarded to the underlying
  transformer/estimator. But before forwarding the incoming data (the joint (X,
  y) datastructure) to the underlying transformer/estimator, the data is split
  into X and y and the split data is passed on as separate arguments. In order
  to filter outliers or other data-points from training but not during test, we
  can create a transformer that implements the filtering logic in its
  `fit_transform` method. This transformer will not be couched within the
  `XyAdapter` and will receive the joint (X, y) datastructure as a whole. Any
  row-filtering operations it performs will be performed on both the features
  and labels.
- A modified `ColumnTransformer` implementation that does not relinquish
  responsibility for all columns that are fed to one or more transformer
  pipelines. Introduces a new `get_feature_names_used` mechanism for
  transformers to let `ColumnTransformer` know which of the input columns it
  wants to be responsible for. Only columns that the transformer takes
  responsibility for are removed from the remainder list. For all other columns
  that were provided to the transformer, responsibility continues to rest with
  `ColumnTransformer` and these columns are not removed from the 'remainder'
  mechanism. If a transformer does not implement the `get_feature_names_used`
  method, then we assume that the transformer has taken responsibility for all
  the columns that were sent to it, which defaults to the current behaviour in
  `ColumnTransformer`.

## Example workflow

### Note implementing new features being cumbersome

In this note, we provide more details on why it is cumbersome to create new
features while retaining original features with their original feature names.

Prefixing can be messy and lead to duplicate columns when the original features
need to be retained. This is because `ColumnTransformer` assumes that if a
column is used to feed one or more transformer pipelines then these columns are
'owned' by the pipelines and surfacing them again if needed is the pipeline's
responsibility. But this problematic when a could is fed to multiple pipelines.
Avoid duplicates couples the pipelines together resulting in bad design.
Alternatively, these original columns can be resurfaced using the remainder
passthrough mechanism without introducing coupling in the individual
transformers that are used in `ColumnTransformer`. But 'used' columns are
removed In `ColumnTransformer`, these columns are then removed from the
'remainder' column list and thus cannot be surfaced . These used-up columns can
no longer be passthrough-ed using the remainder='passthrough' mechanism.
