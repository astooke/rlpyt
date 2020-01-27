
import numpy as np
import multiprocessing as mp
import ctypes
import torch

from rlpyt.utils.collections import namedarraytuple_like
# from rlpyt.utils.misc import put


def buffer_from_example(example, leading_dims, share_memory=False):
    """Allocates memory and returns it in `namedarraytuple` with same
    structure as ``examples``, which should be a `namedtuple` or
    `namedarraytuple`. Applies the same leading dimensions ``leading_dims`` to
    every entry, and otherwise matches their shapes and dtypes. The examples
    should have no leading dimensions.  ``None`` fields will stay ``None``.
    Optionally allocate on OS shared memory. Uses ``build_array()``.
    """
    if example is None:
        return
    try:
        buffer_type = namedarraytuple_like(example)
    except TypeError:  # example was not a namedtuple or namedarraytuple
        return build_array(example, leading_dims, share_memory)
    return buffer_type(*(buffer_from_example(v, leading_dims, share_memory)
        for v in example))


def build_array(example, leading_dims, share_memory=False):
    """Allocate a numpy array matchin the dtype and shape of example, possibly
    with additional leading dimensions.  Optionally allocate on OS shared
    memory.
    """
    a = np.asarray(example)
    if a.dtype == "object":
        raise TypeError("Buffer example value cannot cast as np.dtype==object.")
    constructor = np_mp_array if share_memory else np.zeros
    if not isinstance(leading_dims, (list, tuple)):
        leading_dims = (leading_dims,)
    return constructor(shape=leading_dims + a.shape, dtype=a.dtype)


def np_mp_array(shape, dtype):
    """Allocate a numpy array on OS shared memory."""
    size = int(np.prod(shape))
    nbytes = size * np.dtype(dtype).itemsize
    mp_array = mp.RawArray(ctypes.c_char, nbytes)
    return np.frombuffer(mp_array, dtype=dtype, count=size).reshape(shape)


def torchify_buffer(buffer_):
    """Convert contents of ``buffer_`` from numpy arrays to torch tensors.
    ``buffer_`` can be an arbitrary structure of tuples, namedtuples, and
    namedarraytuples, and a new, matching structure will be returned.
    ``None`` fields remain ``None``, and torch tensors are left alone."""
    if buffer_ is None:
        return
    if isinstance(buffer_, np.ndarray):
        return torch.from_numpy(buffer_)
    elif isinstance(buffer_, torch.Tensor):
        return buffer_
    contents = tuple(torchify_buffer(b) for b in buffer_)
    if type(buffer_) is tuple:  # tuple, namedtuple instantiate differently.
        return contents
    return type(buffer_)(*contents)


def numpify_buffer(buffer_):
    """Convert contents of ``buffer_`` from torch tensors to numpy arrays.
    ``buffer_`` can be an arbitrary structure of tuples, namedtuples, and
    namedarraytuples, and a new, matching structure will be returned.
    ``None`` fields remain ``None``, and numpy arrays are left alone."""
    if buffer_ is None:
        return
    if isinstance(buffer_, torch.Tensor):
        return buffer_.cpu().numpy()
    elif isinstance(buffer_, np.ndarray):
        return buffer_
    contents = tuple(numpify_buffer(b) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def buffer_to(buffer_, device=None):
    """Send contents of ``buffer_`` to specified device (contents must be
    torch tensors.). ``buffer_`` can be an arbitrary structure of tuples,
    namedtuples, and namedarraytuples, and a new, matching structure will be
    returned."""
    if buffer_ is None:
        return
    if isinstance(buffer_, torch.Tensor):
        return buffer_.to(device)
    elif isinstance(buffer_, np.ndarray):
        raise TypeError("Cannot move numpy array to device.")
    contents = tuple(buffer_to(b, device=device) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def buffer_method(buffer_, method_name, *args, **kwargs):
    """Call method ``method_name(*args, **kwargs)`` on all contents of
    ``buffer_``, and return the results. ``buffer_`` can be an arbitrary
    structure of tuples, namedtuples, and namedarraytuples, and a new,
    matching structure will be returned.  ``None`` fields remain ``None``.
    """
    if buffer_ is None:
        return
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):
        return getattr(buffer_, method_name)(*args, **kwargs)
    contents = tuple(buffer_method(b, method_name, *args, **kwargs) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def buffer_func(buffer_, func, *args, **kwargs):
    """Call function ``func(buf, *args, **kwargs)`` on all contents of
    ``buffer_``, and return the results.  ``buffer_`` can be an arbitrary
    structure of tuples, namedtuples, and namedarraytuples, and a new,
    matching structure will be returned.  ``None`` fields remain ``None``.
    """
    if buffer_ is None:
        return
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):
        return func(buffer_, *args, **kwargs)
    contents = tuple(buffer_func(b, func, *args, **kwargs) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def get_leading_dims(buffer_, n_dim=1):
    """Return the ``n_dim`` number of leading dimensions of the contents of
    ``buffer_``. Checks to make sure the leading dimensions match for all
    tensors/arrays, except ignores ``None`` fields.
    """
    if buffer_ is None:
        return
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):
        return buffer_.shape[:n_dim]
    contents = tuple(get_leading_dims(b, n_dim) for b in buffer_ if b is not None)
    if not len(set(contents)) == 1:
        raise ValueError(f"Found mismatched leading dimensions: {contents}")
    return contents[0]


# def buffer_put(x, loc, y, axis=0, wrap=False):
#     if isinstance(x, (np.ndarray, torch.Tensor)):
#         put(x, loc, y, axis=axis, wrap=wrap)
#     else:
#         for vx, vy in zip(x, y):
#             buffer_put(vx, loc, vy, axis=axis, wrap=wrap)
