
from collections import namedtuple


def tuple_itemgetter(i):
    def _tuple_itemgetter(nat):
        return tuple.__getitem__(nat, i)
    return _tuple_itemgetter


def namedarraytuple(typename, field_names, return_namedtuple_cls=False):
    """
    Returns a new subclass of a namedtuple which allows indexing / slicing
    reads and writes of the contained values, intended to be numpy arrays which
    share some dimensions.
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
        return self.__class__(*(s[loc] for s in self))

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
        """Unlike a tuple, but like a dict, checks presence of field name."""
        return key in self._fields

    def get(self, loc):
        "Retrieve value as if indexing into regular tuple."
        return tuple.__getitem__(self, loc)

    def get_field(self, field_name):
        "Retrieve value by field name."
        if field_name not in self._fields:
            raise KeyError(f"Unrecognized field '{field}'.")
        return getattr(self, field_name)

    for method in (__getitem__, __setitem__, get, get_field):
        method.__qualname__ = f'{typename}.{method.__name__}'

    arg_list = repr(NtCls._fields).replace("'", "")[1:-1]
    class_namespace = {
        '__doc__': f'{typename}({arg_list})',
        '__slots__': (),
        '__getitem__': __getitem__,
        '__setitem__': __setitem__,
        '__contains__': __contains__,
        'get': get,
        'get_field': get_field,
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
