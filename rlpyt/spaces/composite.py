

from rlpyt.spaces.base import Space


class Composite(Space):
    """A space for composing arbitrary combinations of spaces together."""

    def __init__(self, spaces, NamedTupleCls):
        """Must input the instantiated sub-spaces in order (e.g. list or
        tuple), and a named tuple class with whch to organize the sub-spaces
        and resulting samples.  The ``NamedTupleCls`` should be defined in 
        the module (file) which defines the composite space.
        """
        self._spaces = spaces
        # Should define NamedTupleCls in the module creating this space.
        self._NamedTupleCls = NamedTupleCls

    def sample(self):
        """Return a single sample which is a named tuple composed of samples 
        from all sub-spaces."""
        return self._NamedTupleCls(*(s.sample() for s in self._spaces))

    def null_value(self):
        """Return a null value which is a named tuple composed of null
        values from all sub-spaces."""
        return self._NamedTupleCls(*(s.null_value() for s in self._spaces))

    @property
    def shape(self):
        """Return a named tuple composed of shapes of every sub-space."""
        return self._NamedTupleCls(*(s.shape for s in self._spaces))

    @property
    def names(self):
        """Return names of sub-spaces."""
        return self._NamedTupleCls._fields

    @property
    def spaces(self):
        """Return the bare sub-spaces."""
        return self._spaces

    def __repr__(self):
        return ", ".join(space.__repr__() for space in self._spaces)
