

from rlpyt.spaces.base import Space


class Composite(Space):

    def __init__(self, spaces, NamedTupleCls):
        self._spaces = spaces
        # Should define NamedTupleCls in the module creating this space.
        self._NamedTupleCls = NamedTupleCls

    def sample(self):
        return self._NamedTupleCls(*(s.sample() for s in self._spaces))

    def null_value(self):
        return self._NamedTupleCls(*(s.null_value() for s in self._spaces))

    @property
    def shape(self):
        return self._NamedTupleCls(*(s.shape for s in self._spaces))

    @property
    def names(self):
        return self._NamedTupleCls._fields

    @property
    def spaces(self):
        return self._spaces

    def __repr__(self):
        return ", ".join(space.__repr__() for space in self._spaces)
