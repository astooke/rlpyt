
import numpy as np
from gym.spaces.dict import Dict as GymDict
from collections import namedtuple

from rlpyt.utils.collections import is_namedtuple_class, is_namedtuple
from rlpyt.spaces.composite import Composite


class GymSpaceWrapper:
    """Wraps a gym space to match the rlpyt interface; most of
    the functionality is for automatically converting a GymDict (dictionary)
    space into an rlpyt Composite space (and converting between the two).  Use
    inside the initialization of the environment wrapper for a gym environment.
    """

    def __init__(self, space, null_value=0, name="obs", force_float32=True):
        """Input ``space`` is a gym space instance.  
        
        Input ``name`` is used to disambiguate different gym spaces being
        wrapped, which is necessary if more than one GymDict space is to be
        wrapped in the same file.  The reason is that the associated
        namedtuples must be defined in the globals of this file, so they must
        have distinct names.
        """
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

    def sample(self):
        """Returns a single sample in a namedtuple (for composite) or numpy
        array using the the ``sample()`` method of the underlying gym
        space(s)."""
        sample = self.space.sample()
        if self.space is self._gym_space:  # Not Composite.
            # Force numpy array, might force float64->float32.
            sample = np.asarray(sample, dtype=self._dtype)
        return sample

    def null_value(self):
        """Similar to ``sample()`` but returning a null value."""
        if self.space is self._gym_space:
            null = np.asarray(self.space.sample(), dtype=self._dtype)
            if self._null_value is not None:
                try:
                    null[:] = self._null_value
                except IndexError:  # e.g. scalar.
                    null.fill(self._null_value)
            else:
                null.fill(0)
        else:  # Is composite.
            null = self.space.null_value()
        return null

    def convert(self, value):
        """For dictionary space, use to convert wrapped env's dict to rlpyt
        namedtuple, i.e. inside the environment wrapper's ``step()``, for
        observation output to the rlpyt sampler (see helper function in
        file)"""
        return dict_to_nt(value, name=self._base_name)

    def revert(self, value):
        """For dictionary space, use to revert namedtuple action into wrapped
        env's dict, i.e. inside the environment wrappers ``step()``, for input
        to the underlying gym environment (see helper function in file)."""
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

    def seed(self, seed=None):
        return self.space.seed(seed=seed)


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
