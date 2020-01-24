
import numpy as np
import torch


def iterate_mb_idxs(data_length, minibatch_size, shuffle=False):  # , horizon=1)
    """Yields minibatches of indexes, to use as a for-loop iterator, with
    option to shuffle.
    """
    if shuffle:
        indexes = np.arange(data_length)
        np.random.shuffle(indexes)
    for start_idx in range(0, data_length - minibatch_size + 1, minibatch_size):
        batch = slice(start_idx, start_idx + minibatch_size)
        if shuffle:
            batch = indexes[batch]
        yield batch


def zeros(shape, dtype):
    """Attempt to return torch tensor of zeros, or if numpy dtype provided,
    return numpy array or zeros."""
    try:
        return torch.zeros(shape, dtype=dtype)
    except TypeError:
        return np.zeros(shape, dtype=dtype)


def empty(shape, dtype):
    """Attempt to return empty torch tensor, or if numpy dtype provided,
    return empty numpy array."""
    try:
        return torch.empty(shape, dtype=dtype)
    except TypeError:
        return np.empty(shape, dtype=dtype)


def extract_sequences(array_or_tensor, T_idxs, B_idxs, T):
    """Assumes `array_or_tensor` has [T,B] leading dims.  Returns new
    array/tensor which contains sequences of length [T] taken from the
    starting indexes [T_idxs, B_idxs], where T_idxs (and B_idxs) is a list or
    vector of integers. Handles wrapping automatically. (Return shape: [T,
    len(B_idxs),...])."""
    shape = (T, len(B_idxs)) + array_or_tensor.shape[2:]
    sequences = empty(shape, dtype=array_or_tensor.dtype)
    for i, (t, b) in enumerate(zip(T_idxs, B_idxs)):
        if t + T > len(array_or_tensor):  # Wrap end.
            m = len(array_or_tensor) - t
            sequences[:m, i] = array_or_tensor[t:, b]  # [m,..]
            sequences[m:, i] = array_or_tensor[:T - m, b]  # [w,..]
        elif t < 0:  # Wrap beginning.
            sequences[t:, i] = array_or_tensor[t:, b]
            sequences[:t, i] = array_or_tensor[:t + T, b]
        else:
            sequences[:, i] = array_or_tensor[t:t + T, b]  # [T,..]
    return sequences


# def put(x, loc, y, axis=0, wrap=False):
#     """Write x[..,loc,..] = y, with loc at chosen axis; supports wrapping of
#     indexes or slice.  Param x can be torch tensor or numpy array."""
#     slc = [slice(None)] * x.ndim
#     if not wrap:
#         slc[axis] = loc
#         x[tuple(slc)] = y
#         return
#     ax_len = x.shape[axis]
#     if isinstance(loc, slice):
#         if loc.start is None or loc.stop is None:
#             start = None if loc.start is None else loc.start % ax_len
#             stop = None if loc.stop is None else loc.stop % ax_len
#             slc[axis] = slice(start, stop, loc.step)
#             x[tuple(slc)] = y
#         elif loc.start == loc.stop:
#             return
#         else:
#             assert 0 < loc.stop - loc.start <= ax_len
#             start = loc.start % ax_len
#             stop = loc.stop % ax_len
#             if stop > start:
#                 slc[axis] = slice(start, stop, loc.step)
#                 x[tuple(slc)] = y
#             else:
#                 step = loc.step or 1
#                 slc1 = slice(start, None, loc.step)
#                 start2 = (step - ax_len + start) % step  # (0 if step==1)
#                 slc2 = slice(start2, stop, loc.step)
#                 yloc = int(np.ceil((ax_len - start) / step))
#                 slc[axis] = slc1
#                 x[tuple(slc)] = y[:yloc]
#                 slc[axis] = slc2
#                 x[tuple(slc)] = y[yloc:]
#     else:  # isinstance(loc, (int, tuple, list, ndarray, tensor))
#         if isinstance(loc, int):
#             loc = [loc]
#         slc[axis] = [loc_ % ax_len for loc_ in loc]
#         x[tuple(slc)] = y
