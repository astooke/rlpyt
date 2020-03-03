
import torch


def select_at_indexes(indexes, tensor):
    """Returns the contents of ``tensor`` at the multi-dimensional integer
    array ``indexes``. Leading dimensions of ``tensor`` must match the
    dimensions of ``indexes``.
    """
    dim = len(indexes.shape)
    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = tensor.view((num,) + tensor.shape[dim:])
    s_flat = t_flat[torch.arange(num), indexes.view(-1)]
    return s_flat.view(tensor.shape[:dim] + tensor.shape[dim + 1:])


def to_onehot(indexes, num, dtype=None):
    """Converts integer values in multi-dimensional tensor ``indexes``
    to one-hot values of size ``num``; expanded in an additional
    trailing dimension."""
    if dtype is None:
        dtype = indexes.dtype
    onehot = torch.zeros(indexes.shape + (num,),
        dtype=dtype, device=indexes.device)
    onehot.scatter_(-1, indexes.unsqueeze(-1).type(torch.long), 1)
    return onehot


def from_onehot(onehot, dim=-1, dtype=None):
    """Argmax over trailing dimension of tensor ``onehot``. Optional return
    dtype specification."""
    indexes = torch.argmax(onehot, dim=dim)
    if dtype is not None:
        indexes = indexes.type(dtype)
    return indexes


def valid_mean(tensor, valid=None, dim=None):
    """Mean of ``tensor``, accounting for optional mask ``valid``,
    optionally along a dimension."""
    dim = () if dim is None else dim
    if valid is None:
        return tensor.mean(dim=dim)
    valid = valid.type(tensor.dtype)  # Convert as needed.
    return (tensor * valid).sum(dim=dim) / valid.sum(dim=dim)


def infer_leading_dims(tensor, dim):
    """Looks for up to two leading dimensions in ``tensor``, before
    the data dimensions, of which there are assumed to be ``dim`` number.
    For use at beginning of model's ``forward()`` method, which should 
    finish with ``restore_leading_dims()`` (see that function for help.)
    Returns:
    lead_dim: int --number of leading dims found.
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    shape: tensor shape after leading dims.
    """
    lead_dim = tensor.dim() - dim
    assert lead_dim in (0, 1, 2)
    if lead_dim == 2:
        T, B = tensor.shape[:2]
    else:
        T = 1
        B = 1 if lead_dim == 0 else tensor.shape[0]
    shape = tensor.shape[lead_dim:]
    return lead_dim, T, B, shape


def restore_leading_dims(tensors, lead_dim, T=1, B=1):
    """Reshapes ``tensors`` (one or `tuple`, `list`) to to have ``lead_dim``
    leading dimensions, which will become [], [B], or [T,B].  Assumes input
    tensors already have a leading Batch dimension, which might need to be
    removed. (Typically the last layer of model will compute with leading
    batch dimension.)  For use in model ``forward()`` method, so that output
    dimensions match input dimensions, and the same model can be used for any
    such case.  Use with outputs from ``infer_leading_dims()``."""
    is_seq = isinstance(tensors, (tuple, list))
    tensors = tensors if is_seq else (tensors,)
    if lead_dim == 2:  # (Put T dim.)
        tensors = tuple(t.view((T, B) + t.shape[1:]) for t in tensors)
    if lead_dim == 0:  # (Remove B=1 dim.)
        assert B == 1
        tensors = tuple(t.squeeze(0) for t in tensors)
    return tensors if is_seq else tensors[0]
