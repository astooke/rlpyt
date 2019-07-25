
import numpy as np
from gym.spaces.dict import Dict as GymDict
from collections import namedtuple

from rlpyt.utils.collection import is_namedtuple_class, is_namedtuple
from rlpyt.spaces.composite import Composite


class GymSpaceWrapper(object):
    """Wraps a gym space to interface from dictionaries to namedtuples."""

    def __init__(self, space, null_value=0, name="obs", force_float32=True):
        self._gym_space = space
        self._base_name = name
        self._null_value = null_value
        if isinstance(space, GymDict):
            nt = globals().get(name)
            if nt is None:
                nt = namedtuple(name, [k for k in space.spaces.keys()])
                globals()[name] = nt  # Put at module level for pickle.
            elif not (is_namedtuple_class(nt) and
                    sorted(nt._fields) ==
                    sorted([k for k in space.spaces.keys()])):
                raise ValueError(f"Name clash in globals: {name}.")
            spaces = [GymSpaceWrapper(
                space=v,
                null_value=null_value,
                name="_".join([name, k]),
                force_float32=force_float32)
                for k, v in space.spaces.items()]
            self.space = Composite(spaces, nt)
            self._dtype = None
        else:
            self.space = space
            self._dtype = np.float32 if (space.dtype == np.float64 and
                force_float32) else None

    def sample(self, null=False):
        if self.space is self._gym_space:
            sample = self.space.sample()
            if self._dtype is not None:  # Might force float64->float32.
                sample = np.asarray(sample, dtype=self._dtype)
            if null and self._null_value is not None:
                sample[:] = self._null_value
        else:  # is Composite
            sample = self.space.sample(null=null)
        return sample

    def convert(self, value):
        # Convert wrapped env's observation from dict to namedtuple.
        return dict_to_nt(value, name=self._base_name)

    def revert(self, value):
        # Revert namedtuple action into wrapped env's dict.
        return nt_to_dict(value)

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

    @property
    def n(self):
        return self.space.n


def dict_to_nt(value, name):
    if isinstance(value, dict):
        values = {k: dict_to_nt(v, "_".join([name, k]))
            for k, v in value.items()}
        return globals()[name](**values)
    if isinstance(value, np.ndarray) and value.dtype == np.float64:
        return np.asarray(value, dtype=np.float32)
    return value


def nt_to_dict(value):
    if is_namedtuple(value):
        return {k: nt_to_dict(v) for k, v in zip(value._fields, value)}
    return value
