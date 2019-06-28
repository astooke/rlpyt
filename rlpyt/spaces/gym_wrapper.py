
import numpy as np


class SpaceWrapper(object):
    """Wraps a gym space to enable multiple values to be sampled at once."""

    def __init__(self, space, dtype=None):
        self.space = space
        self._null_value = 0
        self._dtype = dtype

    def set_null_value(self, null_value):
        self._null_value = null_value

    def sample(self, size=None, null=False):
        """Enable multiple samples to be returned at once."""
        if size is None:
            sample = self.space.sample()
        elif isinstance(size, int):
            sample = np.stack([self.space.sample() for _ in range(size)])
        else:  # isinstance(size, (list, tuple))
            sample = np.stack([self.space.sample()
                for _ in range(int(np.prod(size)))]).reshape(*size, -1)
        if null:
            sample[:] = self._null_value
        return sample

    @property
    def dtype(self):
        return self._dtype or self.space.dtype

    @property
    def shape(self):
        return self.space.shape

    def contains(self, x):
        return self.space.contains(x)

    def __repr__(self):
        return self.space.__repr__()

    def __eq__(self, other):
        return self.space.__eq__(other)

    @property
    def low(self):
        return self.space.low

    @property
    def high(self):
        return self.space.high
