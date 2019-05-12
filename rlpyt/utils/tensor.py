
import numpy as np
import torch


def select_at_indexes(indexes, array_or_tensor):
    """Leading dimensions of array_or_tensor must match dimensions of indexes."""
    dim = len(indexes.shape)
    assert indexes.shape == array_or_tensor.shape[:dim]
    num = int(np.prod(indexes.shape))
    if isinstance(array_or_tensor, np.ndarray):
        a = array_or_tensor
        a_flat = a.reshape(num, *a.shape[dim:])
        s_flat = a_flat[np.arange(num), indexes.reshape(-1)]
        selected = s_flat.reshape(*a.shape[:dim], *a.shape[dim + 1:])
    else:  # torch.Tensor
        t = array_or_tensor
        t_flat = t.view(num, *t.shape[dim:])
        s_flat = t_flat[torch.arange(num), indexes.view(-1)]
        selected = s_flat.view(*t.shape[:dim], *t.shape[dim + 1:])
    return selected


def to_onehot(indexes, dim, dtype=None):
    if dtype is None:
        dtype = indexes.dtype
    if isinstance(indexes, np.ndarray):
        onehot = np.zeros((indexes.size, dim), dtype=dtype)
        onehot[np.arange(indexes.size), indexes.reshape(-1)] = 1
        onehot = onehot.reshape(indexes.shape + (dim,))
    else:  # torch.Tensor
        onehot = torch.zeros((indexes.numel(), dim), dtype=dtype)
        onehot[torch.arange(indexes.numel()), indexes.view(-1)] = 1
        onehot = onehot.view(indexes.shape + (dim,))
    return onehot


def from_onehot(onehot, dtype=None):
    if isinstance(onehot, np.ndarray):
        indexes = np.asarray(np.argmax(onehot, axis=-1), dtype=dtype)
    else:  # torch.Tensor
        indexes = torch.argmax(onehot, axis=-1)
        if dtype is not None:
            indexes = indexes.type(dtype)
    return indexes


def valids_mean(array_or_tensor, valids):
    return (array_or_tensor * valids).sum() / valids.sum()


def unsqueeze_nat(nat, dim=0):
    if isinstance(nat, torch.Tensor):
        return nat.unsqueeze(dim)
    else:
        return type(nat)(unsqueeze_nat(n, dim) for n in nat)


def squeeze_nat(nat, dim=None):
    if isinstance(nat, torch.Tensor):
        return nat.squeeze(dim)
    else:
        return type(nat)(unsqueeze_nat(n, dim) for n in nat)
