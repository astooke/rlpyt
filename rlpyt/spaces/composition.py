

from rlpyt.spaces.base import Space


class Composition(Space):

    def __init__(self, spaces, NamedArrayTupleCls):
        self._spaces = spaces
        # Should define NamedArrayTupleCls in the module creating this space.
        self._NamedArrayTupleCls = NamedArrayTupleCls

    def sample(self, size, null=False, torchify=False):
        return self._NamedArrayTupleCls(*(space.sample(size, null, torchify)
            for space in self._spaces))

    @property
    def shape(self):
        # Could just be namedtuple.
        return self._NamedArrayTupleCls(*(space.shape for space in self._spaces))

    @property
    def names(self):
        return self._NamedArrayTupleCls._fields

    @property
    def spaces(self):
        return self._spaces

    @property
    def null_value(self):
        return self._NamedArrayTupleCls(*(space.null_value for space in self._spaces))

    def __repr__(self):
        return ", ".join(space.__repr__() for space in self._spaces)
