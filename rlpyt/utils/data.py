

from collections import namedtuple
from builtins import property as _property
from functools import partial


def namedarraytuple(typename, fields):

    nt_cls = namedtuple(typename, fields)

    class NAT(object):

        _fields = nt_cls._fields

        def __init__(self, *args, **kwargs):
            self._namedtuple = nt_cls(*args, **kwargs)

        def __getitem__(self, i):
            return self.__class__(*(s[i] for s in self))

        def __setitem__(self, i, val):
            for j, v in enumerate(val):
                self._namedtuple[j][i] = v

        def __iter__(self):
            return self._namedtuple.__iter__()

        def __repr__(self):
            return self._namedtuple.__repr__()

        def __len__(self):
            return self._namedtuple.__len__()

    for field in nt_cls._fields:
        setattr(NAT, field, _property(partial(lambda self, f:
            getattr(self._namedtuple, f), f=field)))

    return NAT






class RlData(object):
    """
    Class that behaves partly like a dictionary for key-indexing and
    like a numpy array for int and slice-indexing.
    Slice-indexing will apply to each field.

    Purpose is to allow the code to stay the same whether observations is
    one or more numpy arrays, i.e. observations[s] = obs always, and
    never need for o in obs.items(): observations[k][s] = o
    """

    def __init__(self, **kwargs):
        self.__dict__ = dict(**kwargs)

    def __getitem__(self, i):
        return self.__class__(**{k: v[i] for k, v in self.__dict__.items()})

    def __setitem__(self, i, val):
        for k, v in val.items():
            self.__dict__[k][i] = v

    def __repr__(self):
        return self.__dict__.__repr__()

    def items(self):
        for k, v in self.__dict__.items():
            yield k, v

    def values(self):
        for v in self.__dict__.values():
            yield v

    def keys(self):
        for k in self.__dict__.keys():
            yield k

    # def __len__(self):
    #     return len(self._data)

    # @property
    # def dtype(self):
    #     """Returns data type."""
    #     return self._dtype

    # @property
    # def ndim(self):
    #     """Returns number of dimensions."""
    #     return self._data.ndim

    # @property
    # def shape(self):
    #     """Returns shape of underlying numpy data array."""
    #     return self._data.shape

    # @property
    # def size(self):
    #     """Returns size of underlying numpy data array."""
    #     return self._data.size

    # @property
    # def data(self):
    #     """Returns underlying numpy data array.  In general, it is not
    #     recommended to manipulate this object directly, aside from reading
    #     from or writing to it (without changing shape).  It may be passed 
    #     to other python processes and used as shared memory.
    #     """
    #     return self._data

    # @property
    # def alloc_size(self):
    #     """Returns the size of the underlying memory allocation (units: number
    #     of items).  This may be larger than the size of the underlying numpy
    #     array, which may occupy only a portion of the allocation (always
    #     starting at the same memory address as the allocation).
    #     """
    #     return self._alloc_size

    # @property
    # def name(self):
    #     """Returns the name (may be None)"""
    #     return self._name