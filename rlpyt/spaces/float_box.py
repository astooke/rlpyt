
import numpy as np

from rlpyt.spaces.base import Space


class FloatBox(Space):
    """A box in `R^n`, with specifiable bounds and dtype."""

    def __init__(self, low, high, shape=None, null_value=0., dtype="float32"):
        """
        Two kinds of valid input:
            * low and high are scalars, and shape is provided: Box(-1.0, 1.0, (3,4))
            * low and high are arrays of the same shape: Box(np.array([-1.0,-2.0]), np.array([2.0,4.0]))
        """
        self.dtype = np.dtype(dtype)
        assert np.issubdtype(self.dtype, np.floating)
        if shape is None:
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            assert self.low.shape == self.high.shape
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = np.asarray(low + np.zeros(shape), dtype=dtype)
            self.high = np.asarray(high + np.zeros(shape), dtype=dtype)
        self._null_value = null_value

    def sample(self):
        """Return a single sample from ``np.random.uniform``."""
        return np.asarray(np.random.uniform(low=self.low, high=self.high,
            size=self.shape), dtype=self.dtype)

    def null_value(self):
        null = np.zeros(self.shape, dtype=self.dtype)
        if self._null_value is not None:
            try:
                null[:] = self._null_value
            except IndexError:
                null.fill(self._null_value)
        return null

    @property
    def shape(self):
        return self.low.shape

    @property
    def bounds(self):
        return self.low, self.high

    def __repr__(self):
        return f"FloatBox{self.shape}"
