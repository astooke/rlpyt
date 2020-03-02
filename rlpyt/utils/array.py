
import numpy as np


def select_at_indexes(indexes, array):
    """Returns the contents of ``array`` at the multi-dimensional integer
    array ``indexes``. Leading dimensions of ``array`` must match the
    dimensions of ``indexes``.
    """
    dim = len(indexes.shape)
    assert indexes.shape == array.shape[:dim]
    num = int(np.prod(indexes.shape))
    a_flat = array.reshape((num,) + array.shape[dim:])
    s_flat = a_flat[np.arange(num), indexes.reshape(-1)]
    selected = s_flat.reshape(array.shape[:dim] + array.shape[dim + 1:])
    return selected


def to_onehot(indexes, dim, dtype=None):
    """Converts integer values in multi-dimensional array ``indexes``
    to one-hot values of size ``dim``; expanded in an additional
    trailing dimension."""
    dtype = indexes.dtype if dtype is None else dtype
    onehot = np.zeros((indexes.size, dim), dtype=dtype)
    onehot[np.arange(indexes.size), indexes.reshape(-1)] = 1
    return onehot.reshape(indexes.shape + (dim,))


def from_onehot(onehot, dtype=None):
    """Argmax over trailing dimension of array ``onehot``. Optional return
    dtype specification."""
    return np.asarray(np.argmax(onehot, axis=-1), dtype=dtype)


def valid_mean(array, valid=None, axis=None):
    """Mean of ``array``, accounting for optional mask ``valid``,
    optionally along an axis."""
    if valid is None:
        return array.mean(axis=axis)
    return (array * valid).sum(axis=axis) / valid.sum(axis=axis)


def infer_leading_dims(array, dim):
    """Determine any leading dimensions of ``array``, which can have up to two
    leading dimensions more than the number of data dimensions, ``dim``.  Used
    to check for [B] or [T,B] leading.  Returns size of leading dimensions (or
    1 if they don't exist), the data shape, and whether the leading dimensions
    where found.
    """
    assert array.ndim in (dim, dim + 1, dim + 2)
    shape = array.shape[len(array.shape) - dim:]
    T = B = 1
    has_T = has_B = False
    if array.ndim == dim + 2:
        T, B = array.shape[:2]
        has_T = has_B = True  # Might have T=1 or B=1.
    elif array.ndim == dim + 1:
        B = array.shape[0]
        has_B = True
    return T, B, shape, has_T, has_B
