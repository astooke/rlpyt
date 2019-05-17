
import numpy as np
import torch

from rlpyt.spaces.base import Space


class IntBox(Space):
    """A box in J^n, with specificiable bound and dtype."""

    def __init__(self, low, high, shape=None, dtype="int32", null_value=None):
        """
        low and high are scalars, applied across all dimensions of shape.
        """
        assert np.isscalar(low) and np.isscalar(high)
        self.low = low
        self.high = high
        self.shape = shape if shape is not None else ()  # np.ndarray sample
        self.dtype = np.dtype(dtype)
        assert np.issubdtype(self.dtype, np.integer)
        null_value = low if null_value is None else null_value
        assert null_value >= low and null_value < high
        self.null_value = null_value

    def sample(self, size=None, null=False, torchify=False):
        if size is None:
            size = ()
        elif isinstance(size, int):
            size = (size,)
        sample = np.random.randint(low=self.low, high=self.high,
            size=size + self.shape, dtype=self.dtype)
        if null:
            sample[:] = self.null_value
        if torchify:
            sample = torch.from_numpy(sample)
        return sample

    @property
    def bounds(self):
        return self.low, self.high

    @property
    def n(self):
        return self.high - self.low - 1

    def __repr__(self):
        return f"IntBox({self.low}-{self.high - 1} {self.shape})"
