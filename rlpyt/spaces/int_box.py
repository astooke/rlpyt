
import numpy as np
import torch

from rlpyt.spaces.base import Space


class IntBox(Space):
    """A box in J^n, with specificiable bound and dtype."""

    def __init__(self, low, high, shape, dtype="int32"):
        """
        low and high are scalars, applied across all dimensions of shape.
        """
        assert np.isscalar(low) and np.isscalar(high)
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = np.dtype(dtype)
        assert np.issubdtype(self.dtype, np.integer)

    def sample(self, size=None, null=False, torchify=False):
        if size is None:
            size = ()
        elif isinstance(size, int):
            size = (size,)
        if null:
            raise NotImplementedError
        sample = np.random.randint(low=self.low, high=self.high,
            size=size + self.shape, dtype=self.dtype)
        if torchify:
            sample = torch.from_numpy(sample)
        return sample

    @property
    def bounds(self):
        return self.low, self.high

    def __repr__(self):
        return f"IntBox{self.shape}"
