
import torch

from rlpyt.utils.tensor import to_onehot, from_onehot


class DiscreteMixin(object):

    def __init__(self, dim, dtype=torch.long, onehot_dtype=torch.float):
        self._dim = dim
        self.dtype = dtype
        self.onehot_dtype = onehot_dtype

    @property
    def dim(self):
        return self._dim

    def to_onehot(self, indexes, dtype=None):
        dtype = self.onehot_dtype if dtype is None else dtype
        return to_onehot(indexes, self._dim, dtype=dtype)

    def from_onehot(self, onehot, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        return from_onehot(onehot, dtpye=dtype)
