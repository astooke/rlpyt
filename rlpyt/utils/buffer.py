
import numpy as np
import multiprocessing as mp
import ctypes
import torch

from rlpyt.utils.collections import namedarraytuple_like


def buffer_from_example(example, leading_dims, shared_memory=False):
    try:
        buffer_type = namedarraytuple_like(example)
    except TypeError:  # example was not a namedtuple or namedarraytuple
        return build_array(example, leading_dims, shared_memory)
    return buffer_type(*(buffer_from_example(v, leading_dims, shared_memory)
        for v in example))


def build_array(example, leading_dims, shared_memory=False):
    a = np.asarray(example)
    if a.dtype == "object":
        raise TypeError("Buffer example value cannot cast as np.dtype==object.")
    constructor = np_mp_array if shared_memory else np.zeros
    return constructor(shape=leading_dims + a.shape, dtype=a.dtype)


def np_mp_array(shape, dtype):
    size = int(np.prod(shape))
    nbytes = size * np.dtype(dtype).itemsize
    mp_array = mp.RawArray(ctypes.c_char, nbytes)
    return np.frombuffer(mp_array, dtype=dtype, count=size).reshape(shape)


def torchify_buffer(buffer_):
    if isinstance(buffer_, np.ndarray):
        return torch.from_numpy(buffer_)
    elif isinstance(buffer_, torch.Tensor):
        return buffer_
    contents = tuple(torchify_buffer(b) for b in buffer_)
    if type(buffer_) is tuple:  # tuple, namedtuple instantiate differently.
        return contents
    return type(buffer_)(*contents)


def numpify_buffer(buffer_):
    if isinstance(buffer_, torch.Tensor):
        return buffer_.numpy()
    elif isinstance(buffer_, np.ndarray):
        return buffer_
    contents = tuple(numpify_buffer(b) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def buffer_to(buffer_, device=None):
    if isinstance(buffer_, torch.Tensor):
        return buffer_.to(device)
    elif isinstance(buffer_, np.ndarray):
        raise TypeError("Cannot move numpy array to device.")
    contents = tuple(buffer_to(b, device=device) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def buffer_method(buffer_, method_name, *args, **kwargs):
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):
        return getattr(buffer_, method_name)(*args, **kwargs)
    contents = tuple(buffer_method(b, method_name, *args, **kwargs) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def _recurse_buffer(func, buffer_):
    contents = tuple(func(b) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)
