
import numpy as np

from rlpyt.spaces.base import Space


class IntBox(Space):
    """A box in `J^n`, with specificiable bound and dtype."""

    def __init__(self, low, high, shape=None, dtype="int32", null_value=None):
        """
        Params ``low`` and ``high`` are scalars, applied across all dimensions
        of shape; valid values will be those in ``range(low, high)``.
        """
        assert np.isscalar(low) and np.isscalar(high)
        self.low = low
        self.high = high
        self.shape = shape if shape is not None else ()  # np.ndarray sample
        self.dtype = np.dtype(dtype)
        assert np.issubdtype(self.dtype, np.integer)
        null_value = low if null_value is None else null_value
        assert null_value >= low and null_value < high
        self._null_value = null_value

    def sample(self):
        """Return a single sample from ``np.random.randint``."""
        return np.random.randint(low=self.low, high=self.high,
            size=self.shape, dtype=self.dtype)

    def null_value(self):
        null = np.zeros(self.shape, dtype=self.dtype)
        if self._null_value is not None:
            try:
                null[:] = self._null_value
            except IndexError:
                null.fill(self._null_value)
        return null

    @property
    def bounds(self):
        return self.low, self.high

    @property
    def n(self):
        """The number of elements in the space."""
        return self.high - self.low

    def __repr__(self):
        return f"IntBox({self.low}-{self.high - 1} shape={self.shape})"
