
import numpy as np

from rlpyt.spaces.base import Space


class FloatBox(Space):
    """A box in R^n, with specifiable bounds and dtype."""

    def __init__(self, low, high, shape=None, dtype="float32"):
        """
        Two kinds of valid input:
            # low and high are scalars, and shape is provided
            Box(-1.0, 1.0, (3,4))
            # low and high are arrays of the same shape
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0]))
        """
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)
        self.dtype = np.dtype(dtype)
        assert np.issubdtype(self.dtype, np.floating)

    def sample(self, size=None, null=False):
        if size is None:
            size = ()
        elif isinstance(size, int):
            size = (size,)

        if null:
            raise NotImplementedError
        return np.asarray(np.random.uniform(low=self.low, high=self.high,
            size=size + self.shape), dtype=self.dtype)

    @property
    def shape(self):
        return self.low.shape

    @property
    def bounds(self):
        return self.low, self.high

    def __repr__(self):
        return f"FloatBox{self.shape}"
