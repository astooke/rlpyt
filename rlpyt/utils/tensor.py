
import torch


def select_at_indexes(indexes, tensor):
    """Leading dimensions of tensor must match dimensions of indexes."""
    dim = len(indexes.shape)
    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = tensor.view((num,) + tensor.shape[dim:])
    s_flat = t_flat[torch.arange(num), indexes.view(-1)]
    return s_flat.view(tensor.shape[:dim] + tensor.shape[dim + 1:])


def to_onehot(indexes, num, dtype=None):
    """Dimension of size num added to the end of indexes.shape."""
    if dtype is None:
        dtype = indexes.dtype
    onehot = torch.zeros(indexes.shape + (num,),
        dtype=dtype, device=indexes.device)
    onehot.scatter_(-1, indexes.unsqueeze(-1).type(torch.long), 1)
    return onehot


def from_onehot(onehot, dim=-1, dtype=None):
    """Selected dimension of onehot is removed by argmax."""
    indexes = torch.argmax(onehot, dim=dim)
    if dtype is not None:
        indexes = indexes.type(dtype)
    return indexes


def valid_mean(tensor, valid=None):
    if valid is None:
        return tensor.mean()
    return (tensor * valid).sum() / valid.sum()


def infer_leading_dims(tensor, dim):
    """Param 'dim': number of non-leading dimensions in tensor.
    Returns:
    shape: tensor shape after leading dims
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    has_T: boolean --presense of T dim (2 leading)
    has_B: boolean --presence of B dim (at least 1 leading)
    """
    assert tensor.dim() in (dim, dim + 1, dim + 2)
    shape = tensor.shape[-dim:]
    T = B = 1
    has_T = has_B = False
    if tensor.dim() == dim + 2:
        T, B = tensor.shape[:2]
        has_T = has_B = True  # Might have T=1 or B=1.
    elif tensor.dim() == dim + 1:
        B = tensor.shape[0]
        has_B = True
    return shape, T, B, has_T, has_B


def restore_leading_dims(tensors, T=1, B=1, put_T=False, put_B=False):
    """Assume tensors have leading Batch dimension (might need removed)."""
    is_seq = True
    if not isinstance(tensors, (tuple, list)):
        tensors = (tensors,)
        is_seq = False
    if put_T:
        return tuple(t.view((T, B) + t.shape[1:]) for t in tensors)
    if not put_B:  # Assume B=1 leading dim.
        return tuple(t.squeeze(0) for t in tensors)
    return tensors if is_seq else tensors[0]
