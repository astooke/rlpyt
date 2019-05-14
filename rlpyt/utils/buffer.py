
import numpy as np
import multiprocessing as mp
import ctypes
import torch


def buffer_from_example(example, leading_dims, shared_memory=False):
    if hasattr(example, "_fields"):  # Then assume namedtuple; recurse.
        buffer_ = type(example)(*(
            buffer_from_example(v, leading_dims, shared_memory)
            for v in example))
    else:
        buffer_ = build_array(example, leading_dims, shared_memory)
    return buffer_


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
    return type(buffer_)(*(torchify_buffer(b) for b in buffer_))


def numpify_buffer(buffer_):
    if isinstance(buffer_, torch.Tensor):
        return buffer_.numpy()
    elif isinstance(buffer_, np.ndarray):
        return buffer_
    return type(buffer_)(*(numpify_buffer(b) for b in buffer_))


def buffer_to(buffer_, device=None):
    if isinstance(buffer_, torch.Tensor):
        return buffer_.to(device)
    return type(buffer_)(*(buffer_to(b) for b in buffer_))


def buffer_method(buffer_, method_name, *args, **kwargs):
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):
        return getattr(buffer_, method_name)(*args, **kwargs)
    return type(buffer_)(*(buffer_method(b, method_name, *args, **kwargs)
        for b in buffer_))
