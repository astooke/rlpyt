
import numpy as np
import torch

from rlpyt.spaces.base import Space
from rlpyt.utils import tensor


class Discrete(Space):
    """ {0,1,...,n-1}"""

    def __init__(self, n, dtype="int32", onehot_dtype=None, null_value=0):
        self.n = n
        # self._items = np.arange(n)
        self.dtype = np.dtype(dtype)
        self.onehot_dtype = self.dtype if onehot_dtype is None else onehot_dtype
        self.torch_dtype = torch.from_numpy(np.zeros(1, dtype=self.dtype)).dtype
        self.torch_onehot_dtype = torch.from_numpy(
            np.zeros(1, dtype=self.onehot_dtype)).dtype
        assert np.issubdtype(self.dtype, np.integer)
        assert np.issubdtype(self.onhot_dtype, np.integer)
        self.null_value = null_value

    def sample(self, size=None, null=False):
        sample = np.random.randint(low=0, high=self.n, size=size, dtype=dtype)
        if null:
            sample.fill(null_value)
        return sample

    def __repr__(self):
        return f"Discrete({self.n})"

    def to_onehot(self, indexes):
        dtype = self.onehot_dtype if isinstance(indexes, np.ndarray) \
            else self.torch_onehot_dtype
        return tensor.to_onehot(indexes, self.n, dtype=dtype)

    def from_onehot(self, onehot):
        dtype = self.dtype if isinstance(indexes, np.ndarray) \
            else self.torch_dtype
        return tensor.from_onehot(onehot, dtype=dtype)

    def weighted_sample(self):
        # Maybe not needed, because torch.multinomial inside sample_actions().
        raise NotImplementedError
