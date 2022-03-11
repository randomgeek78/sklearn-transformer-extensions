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

In this example, we create a end-to-end example using the AMES housing dataset. 

```python
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
print(DATA_URL)
```

##
