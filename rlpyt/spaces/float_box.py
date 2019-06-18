
import numpy as np
import torch

from rlpyt.spaces.base import Space


class FloatBox(Space):
    """A box in R^n, with specifiable bounds and dtype."""

    def __init__(self, low, high, shape=None, null_value=0., dtype="float32"):
        """
        Two kinds of valid input:
            # low and high are scalars, and shape is provided
            Box(-1.0, 1.0, (3,4))
            # low and high are arrays of the same shape
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0]))
        """
        self.dtype = np.dtype(dtype)
        assert np.issubdtype(self.dtype, np.floating)
        if shape is None:
            assert low.shape == high.shape
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = np.asarray(low + np.zeros(shape), dtype=dtype)
            self.high = np.asarray(high + np.zeros(shape), dtype=dtype)
        self._null_value = null_value

    def sample(self, size=None, null=False, torchify=False):
        if size is None:
            size = ()
        elif isinstance(size, int):
            size = (size,)

        if null:
            sample = self._null_value * np.ones(size + self.shape, dtype=self.dtype)
        else:
            sample = np.asarray(np.random.uniform(low=self.low, high=self.high,
                size=size + self.shape), dtype=self.dtype)
        if torchify:
            sample = torch.from_numpy(sample)
        return sample

    @property
    def shape(self):
        return self.low.shape

    @property
    def bounds(self):
        return self.low, self.high

    @property
    def null_value(self):
        return self._null_value

    def __repr__(self):
        return f"FloatBox{self.shape}"
