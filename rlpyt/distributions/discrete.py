
import torch

from rlpyt.utils.tensor import to_onehot, from_onehot


class DiscreteMixin:

    def __init__(self, dim, dtype=torch.long, onehot_dtype=torch.float):
        self._dim = dim
        self.dtype = dtype
        self.onehot_dtype = onehot_dtype

    @property
    def dim(self):
        return self._dim

    def to_onehot(self, indexes, dtype=None):
        return to_onehot(indexes, self._dim, dtype=dtype or self.onehot_dtype)

    def from_onehot(self, onehot, dtype=None):
        return from_onehot(onehot, dtpye=dtype or self.dtype)
