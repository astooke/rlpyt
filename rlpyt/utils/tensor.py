
import torch


def select_at_indexes(indexes, tensor):
    """Leading dimensions of tensor must match dimensions of indexes."""
    dim = len(indexes.shape)
    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = t.view(num, *t.shape[dim:])
    s_flat = t_flat[torch.arange(num), indexes.view(-1)]
    return s_flat.view(*t.shape[:dim], *t.shape[dim + 1:])


def to_onehot(indexes, dim, dtype=None):
    if dtype is None:
        dtype = indexes.dtype
    onehot = torch.zeros(indexes.shape + (dim,),
        dtype=dtype, device=indexes.device)
    onehot.scatter_(-1, indexes.unsqueeze(-1), 1)
    return onehot


def from_onehot(onehot, dtype=None):
    indexes = torch.argmax(onehot, dim=-1)
    if dtype is not None:
        indexes = indexes.type(dtype)
    return indexes


def valid_mean(tensor, valid=None):
    if valid is None:
        return tensor.mean()
    return (tensor * valid).sum() / valid.sum()


def infer_leading_dims(tensor, dim):
    """Param 'dim': number of data dimensions, check for [B] or [T,B] leading."""
    shape = tensor.shape[dim:]
    T = B = 1
    _T = _B = False
    if tensor.dim() == dim + 2:
        T, B = tensor.shape[:2]
        _T = _B = True  # Might have T=1 or B=1.
    elif tensor.dim() == dim + 1:
        B = tensor.shape[0]
        _B = True
    return T, B, shape, _T, _B
