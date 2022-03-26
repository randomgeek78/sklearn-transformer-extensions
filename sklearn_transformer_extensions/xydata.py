from sklearn.utils import _safe_indexing
from sklearn.utils.validation import (
    _make_indexable,
)

class XyData:
    """XyData object holds both the features and labels in the same
    data-structure. It implements a iterator interface as well as an indexing
    interface. Scikit-learn treats it as a numpy array since it provides a
    shape attribute.

    Parameters
    ----------
    X, y: An iterable, pandas dataframe, numpy array
        Both X and y need to be the same length

    Attributes
    ----------
    X, y

    Examples
    --------
    >>> from sklearn_transformer_extensions import XyData
    >>> import numpy as np
    >>> X = np.c_[1, 2, 3].T
    >>> y = np.r_[2, 4, 6]
    >>> Xy = XyData(X, y)
    >>> print(Xy)
    XyData(X=numpy(shape=(3, 1)), y=numpy(shape=(3,)))
    >>> print(Xy.X.shape, Xy.y.shape)
    (3, 1) (3,)
    >>> X_, y_ = Xy
    >>> print(X_.shape, y_.shape)
    (3, 1) (3,)
    >>> Xy_subset = Xy[:2]
    >>> print(Xy_subset)
    XyData(X=numpy(shape=(2, 1)), y=numpy(shape=(2,)))
    >>> print(Xy_subset.X, Xy_subset.y)
    [[1]
     [2]] [2 4]
    """

    def __init__(self, X, y):

        self.X, self.y = _make_indexable((X, y))

    def __getitem__(self, ind):

        X = _safe_indexing(self.X, ind)
        y = _safe_indexing(self.y, ind)

        return XyData(X, y)

    def __len__(self):

        return len(self.X)

    def __iter__(self):

        return iter((self.X, self.y))

    def __repr__(self):

        tx = 'numpy'
        if hasattr(self.X, "iloc"):
            tx = 'pandas'
        sx = self.X.shape

        ty, sy = None, None
        if self.y is not None:
            ty = 'numpy'
            if hasattr(self.y, "iloc"):
                ty = 'pandas'
            sy = self.y.shape

        format = dict(tx=tx, sx=sx, ty=ty, sy=sy)
        name_fmt = self.__class__.__name__
        X_fmt = '{tx}(shape={sx})'.format(**format)
        y_fmt = '{ty}(shape={sy})'.format(**format) if ty is not None else None
        return f'{name_fmt}(X={X_fmt}, y={y_fmt})'

    def __array__(self, dtype=None):
        pass

    @property
    def shape(self):
        return len(self.X), None


