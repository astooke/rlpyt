
from collections import namedtuple


def tuple_itemgetter(i):
    def _tuple_itemgetter(obj):
        return tuple.__getitem__(obj, i)
    return _tuple_itemgetter


def namedarraytuple(typename, field_names, return_namedtuple_cls=False):
    """
    Returns a new subclass of a namedtuple which exposes indexing / slicing
    reads and writes of the contained values, intended to be numpy arrays which
    share leading dimensions.
    (Code follows pattern of collections.namedtuple.)

    >>> PointsCls = namedarraytuple('Points', ['x', 'y'])
    >>> p = PointsCls(np.array([0, 1]), y=np.array([10, 11]))
    >>> p
    Points(x=array([0, 1]), y=array([10, 11]))
    >>> p.x                         # fields accessible by name
    array([0, 1])
    >>> p[0]                        # get location across all fields
    Points(x=0, y=10)               # (location can be index or slice)
    >>> p.get(0)                    # regular tuple-indexing into field
    array([0, 1])
    >>> p.get_field('y')
    array([10, 11])
    >>> x, y = p                    # unpack like a regular tuple
    >>> x
    array([0, 1])
    >>> p[1] = 2                    # assign value to location of all fields
    >>> p
    Points(x=array([0, 2]), y=array([10, 2]))
    >>> p[1] = PointsCls(3, 30)     # assign to location field-by-field
    >>> p
    Points(x=array([0, 3]), y=array([10, 30]))
    >>> 'x' in p                    # check field name instead of object
    True
    """

    NtCls = namedtuple(typename, field_names)

    def __getitem__(self, loc):
        return type(self)(*(s[loc] for s in self))

    __getitem__.__doc__ = (f"Return a new {typename} instance containing the "
        "selected index or slice from each field.")

    def __setitem__(self, loc, value):
        """
        If input value is the same named[array]tuple type, iterate through its
        fields and assign values into selected index or slice of corresponding
        field.  Else, assign whole of value to selected index or slice of
        all fields.
        """
        if not isinstance(value, NtCls):
            value = (value,) * len(self)
        for j, (s, v) in enumerate(zip(self, value)):
            try:
                s[loc] = v
            except (ValueError, IndexError, TypeError) as e:
                raise Exception(f"Occured at item index {j}.") from e

    def __contains__(self, key):
        """Checks presence of field name (unlike tuple; like dict)."""
        return key in self._fields

    def get_index(self, index):
        "Retrieve value as if indexing into regular tuple."
        return tuple.__getitem__(self, index)

    def get(self, field_name, default=None):
        "Retrieve value by field name (like dict.get())."
        if field_name not in self._fields:
            return default
        return getattr(self, field_name)

    def items(self):
        """Iterate like a dict."""
        for k, v in zip(self._fields, self):
            yield k, v

    for method in (__getitem__, __setitem__, get_index, get, items):
        method.__qualname__ = f'{typename}.{method.__name__}'

    arg_list = repr(NtCls._fields).replace("'", "")[1:-1]
    class_namespace = {
        '__doc__': f'{typename}({arg_list})',
        '__slots__': (),
        '__getitem__': __getitem__,
        '__setitem__': __setitem__,
        '__contains__': __contains__,
        'get_index': get_index,
        'get': get,
        'items': items,
    }

    for index, name in enumerate(NtCls._fields):
        itemgetter_object = tuple_itemgetter(index)
        doc = f'Alias for field number {index}'
        class_namespace[name] = property(itemgetter_object, doc=doc)

    result = type(typename, (NtCls,), class_namespace)
    result.__module__ = NtCls.__module__

    if return_namedtuple_cls:
        return result, NtCls
    return result


class AttrDict(dict):
    """
    Behaves like a dictionary but additionally has attribute-style access
    for both read and write.
    e.g. x["key"] and x.key are the same,
    e.g. can iterate using:  for k, v in x.items().
    Can sublcass for specific data classes; must call AttrDict's __init__().
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

    def copy(self):
        """
        Provides a "deep" copy of all unbroken chains of types AttrDict, but
        shallow copies otherwise, (e.g. numpy arrays are NOT copied).
        """
        return type(self)(**{k: v.copy() if isinstance(v, AttrDict) else v
            for k, v in self.items()})
