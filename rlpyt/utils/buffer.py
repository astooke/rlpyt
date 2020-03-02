
import numpy as np
import multiprocessing as mp
import ctypes
import torch

from rlpyt.utils.collections import (NamedArrayTuple, namedarraytuple_like,
    NamedArrayTupleSchema_like, NamedTuple)


def buffer_from_example(example, leading_dims, share_memory=False,
        use_NatSchema=None):
    """Allocates memory and returns it in `namedarraytuple` with same
    structure as ``examples``, which should be a `namedtuple` or
    `namedarraytuple`. Applies the same leading dimensions ``leading_dims`` to
    every entry, and otherwise matches their shapes and dtypes. The examples
    should have no leading dimensions.  ``None`` fields will stay ``None``.
    Optionally allocate on OS shared memory. Uses ``build_array()``.
    
    New: can use NamedArrayTuple types by the `use_NatSchema` flag, which
    may be easier for pickling/unpickling when using spawn instead
    of fork. If use_NatSchema is None, the type of ``example`` will be used to
    infer what type to return (this is the default)
    """
    if example is None:
        return
    if use_NatSchema is None:
        use_NatSchema = isinstance(example, (NamedTuple, NamedArrayTuple))
    try:
        if use_NatSchema:
            buffer_type = NamedArrayTupleSchema_like(example)
        else:
            buffer_type = namedarraytuple_like(example)
    except TypeError:  # example was not a namedtuple or namedarraytuple
        return build_array(example, leading_dims, share_memory)
    return buffer_type(*(buffer_from_example(v, leading_dims,
        share_memory=share_memory, use_NatSchema=use_NatSchema)
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
    if mp.get_start_method() == "spawn":
        return np_mp_array_spawn(shape, dtype)
    size = int(np.prod(shape))
    nbytes = size * np.dtype(dtype).itemsize
    mp_array = mp.RawArray(ctypes.c_char, nbytes)
    return np.frombuffer(mp_array, dtype=dtype, count=size).reshape(shape)


class np_mp_array_spawn(np.ndarray):
    """Shared ndarray for use with multiprocessing's 'spawn' start method.

    This array can be shared between processes by passing it to a `Process`
    init function (or similar). Note that this can only be shared _on process
    startup_; it can't be passed through, e.g., a queue at runtime. Also it
    cannot be pickled outside of multiprocessing's internals."""
    _shmem = None

    def __new__(cls, shape, dtype=None, buffer=None, offset=None, strides=None,
                order=None):
        # init buffer
        if buffer is None:
            assert offset is None
            assert strides is None
            size = int(np.prod(shape))
            nbytes = size * np.dtype(dtype).itemsize
            # this is the part that can be passed between processes
            shmem = mp.RawArray(ctypes.c_char, nbytes)
            offset = 0
        elif isinstance(buffer, ctypes.Array):
            # restoring from a pickle
            shmem = buffer
        else:
            raise ValueError(
                f"{cls.__name__} does not support specifying custom "
                f" buffers, but was given {buffer!r}")

        # init array
        obj = np.ndarray.__new__(cls, shape, dtype=dtype, buffer=shmem,
                                 offset=offset, strides=strides, order=order)
        obj._shmem = shmem

        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self._shmem = obj._shmem

    def __reduce__(self):
        # credit to https://stackoverflow.com/a/53534485 for awful/wonderful
        # __array_interface__ hack
        absolute_offset = self.__array_interface__['data'][0]
        base_address = ctypes.addressof(self._shmem)
        offset = absolute_offset - base_address
        assert offset <= len(self._shmem), (offset, len(self._shmem))
        order = 'FC'[self.flags['C_CONTIGUOUS']]
        # buffer should get pickled by np
        assert self._shmem is not None, \
            "somehow this lost its _shmem reference"
        newargs = (self.shape, self.dtype, self._shmem, offset, self.strides,
                   order)
        return (type(self), newargs)


def torchify_buffer(buffer_):
    """Convert contents of ``buffer_`` from numpy arrays to torch tensors.
    ``buffer_`` can be an arbitrary structure of tuples, namedtuples,
    namedarraytuples, NamedTuples, and NamedArrayTuples, and a new, matching
    structure will be returned. ``None`` fields remain ``None``, and torch
    tensors are left alone."""
    if buffer_ is None:
        return
    if isinstance(buffer_, np.ndarray):
        return torch.from_numpy(buffer_)
    elif isinstance(buffer_, torch.Tensor):
        return buffer_
    contents = tuple(torchify_buffer(b) for b in buffer_)
    if type(buffer_) is tuple:  # tuple, namedtuple instantiate differently.
        return contents
    return buffer_._make(contents)


def numpify_buffer(buffer_):
    """Convert contents of ``buffer_`` from torch tensors to numpy arrays.
    ``buffer_`` can be an arbitrary structure of tuples, namedtuples,
    namedarraytuples, NamedTuples, and NamedArrayTuples, and a new, matching
    structure will be returned. ``None`` fields remain ``None``, and numpy
    arrays are left alone."""
    if buffer_ is None:
        return
    if isinstance(buffer_, torch.Tensor):
        return buffer_.cpu().numpy()
    elif isinstance(buffer_, np.ndarray):
        return buffer_
    contents = tuple(numpify_buffer(b) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return buffer_._make(contents)


def buffer_to(buffer_, device=None):
    """Send contents of ``buffer_`` to specified device (contents must be
    torch tensors.). ``buffer_`` can be an arbitrary structure of tuples,
    namedtuples, namedarraytuples, NamedTuples and NamedArrayTuples, and a
    new, matching structure will be returned."""
    if buffer_ is None:
        return
    if isinstance(buffer_, torch.Tensor):
        return buffer_.to(device)
    elif isinstance(buffer_, np.ndarray):
        raise TypeError("Cannot move numpy array to device.")
    contents = tuple(buffer_to(b, device=device) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return buffer_._make(contents)


def buffer_method(buffer_, method_name, *args, **kwargs):
    """Call method ``method_name(*args, **kwargs)`` on all contents of
    ``buffer_``, and return the results. ``buffer_`` can be an arbitrary
    structure of tuples, namedtuples, namedarraytuples, NamedTuples, and
    NamedArrayTuples, and a new, matching structure will be returned.
    ``None`` fields remain ``None``.
    """
    if buffer_ is None:
        return
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):
        return getattr(buffer_, method_name)(*args, **kwargs)
    contents = tuple(buffer_method(b, method_name, *args, **kwargs) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return buffer_._make(contents)


def buffer_func(buffer_, func, *args, **kwargs):
    """Call function ``func(buf, *args, **kwargs)`` on all contents of
    ``buffer_``, and return the results.  ``buffer_`` can be an arbitrary
    structure of tuples, namedtuples, namedarraytuples, NamedTuples, and
    NamedArrayTuples, and a new, matching structure will be returned.
    ``None`` fields remain ``None``.
    """
    if buffer_ is None:
        return
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):
        return func(buffer_, *args, **kwargs)
    # contents = tuple(buffer_func(b, func, *args, **kwargs) for b in buffer_)
    contents = tuple(buffer_func(b, func, *args, **kwargs) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return buffer_._make(contents)


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
