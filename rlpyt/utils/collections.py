
import sys
from collections import namedtuple


RESERVED_NAMES = ("get", "items")


def tuple_itemgetter(i):
    def _tuple_itemgetter(obj):
        return tuple.__getitem__(obj, i)
    return _tuple_itemgetter


def namedarraytuple(typename, field_names, return_namedtuple_cls=False,
        classname_suffix=False):
    """
    Returns a new subclass of a namedtuple which exposes indexing / slicing
    reads and writes applied to all contained objects, which must share
    indexing (__getitem__) behavior (e.g. numpy arrays).

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
    nt_typename = typename
    if classname_suffix:
        nt_typename += "_nt"  # Helpful to identify which style of tuple.
        typename += "_nat"

    try:  # For pickling, get location where this function was called.
        # NOTE: (pickling might not work for nested class definition.)
        module = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        module = None
    NtCls = namedtuple(nt_typename, field_names, module=module)

    def __getitem__(self, loc):
        try:
            return type(self)(*(None if s is None else s[loc] for s in self))
        except IndexError as e:
            for j, s in enumerate(self):
                if s is None:
                    continue
                try:
                    _ = s[loc]
                except IndexError:
                    raise Exception(f"Occured in {self.__class__} at field "
                        f"'{self._fields[j]}'.") from e

    __getitem__.__doc__ = (f"Return a new {typename} instance containing "
        "the selected index or slice from each field.")

    def __setitem__(self, loc, value):
        """
        If input value is the same named[array]tuple type, iterate through its
        fields and assign values into selected index or slice of corresponding
        field.  Else, assign whole of value to selected index or slice of
        all fields.  Ignore fields that are both None.
        """
        if not (isinstance(value, tuple) and  # Check for matching structure.
                getattr(value, "_fields", None) == self._fields):
            # Repeat value for each but respect any None.
            value = tuple(None if s is None else value for s in self)
        try:
            for j, (s, v) in enumerate(zip(self, value)):
                if s is not None or v is not None:
                    s[loc] = v
        except (ValueError, IndexError, TypeError) as e:
            raise Exception(f"Occured in {self.__class__} at field "
                f"'{self._fields[j]}'.") from e

    def __contains__(self, key):
        "Checks presence of field name (unlike tuple; like dict)."
        return key in self._fields

    def get(self, index):
        "Retrieve value as if indexing into regular tuple."
        return tuple.__getitem__(self, index)

    def items(self):
        "Iterate ordered (field_name, value) pairs (like OrderedDict)."
        for k, v in zip(self._fields, self):
            yield k, v

    for method in (__getitem__, __setitem__, get, items):
        method.__qualname__ = f'{typename}.{method.__name__}'

    arg_list = repr(NtCls._fields).replace("'", "")[1:-1]
    class_namespace = {
        '__doc__': f'{typename}({arg_list})',
        '__slots__': (),
        '__getitem__': __getitem__,
        '__setitem__': __setitem__,
        '__contains__': __contains__,
        'get': get,
        'items': items,
    }

    for index, name in enumerate(NtCls._fields):
        if name in RESERVED_NAMES:
            raise ValueError(f"Disallowed field name: {name}.")
        itemgetter_object = tuple_itemgetter(index)
        doc = f'Alias for field number {index}'
        class_namespace[name] = property(itemgetter_object, doc=doc)

    result = type(typename, (NtCls,), class_namespace)
    result.__module__ = NtCls.__module__

    if return_namedtuple_cls:
        return result, NtCls
    return result


def is_namedtuple_class(obj):
    """Heuristic, might be spoofed.
    Returns False if obj is namedarraytuple class."""
    if type(obj) is not type or obj is type:
        return False
    if len(obj.mro()) != 3:
        return False
    if obj.mro()[1] is not tuple:
        return False
    if not all(hasattr(obj, attr)
            for attr in ["_fields", "_asdict", "_make", "_replace"]):
        return False
    return True


def is_namedarraytuple_class(obj):
    """Heuristic, might be spoofed.
    Returns False if obj is namedtuple class."""
    if type(obj) is not type or obj is type:
        return False
    if len(obj.mro()) != 4:
        return False
    if not is_namedtuple_class(obj.mro()[1]):
        return False
    if not all(hasattr(obj, attr) for attr in RESERVED_NAMES):
        return False
    return True


def is_namedtuple(obj):
    """Heuristic, might be spoofed.
    Returns False if obj is namedarraytuple."""
    return is_namedtuple_class(type(obj))


def is_namedarraytuple(obj):
    """Heuristic, might be spoofed.
    Returns False if obj is namedtuple."""
    return is_namedarraytuple_class(type(obj))


def namedarraytuple_like(namedtuple_or_class, classname_suffix=False):
    """Returns namedarraytuple class with same name and fields as input
    namedtuple or namedarraytuple instance or class.  If input is
    namedarraytuple instance or class, the same class is returned directly."""
    ntc = namedtuple_or_class
    if is_namedtuple(ntc):
        return namedarraytuple(type(ntc).__name__, ntc._fields,
            classname_suffix=classname_suffix)
    elif is_namedtuple_class(ntc):
        return namedarraytuple(ntc.__name__, ntc._fields,
            classname_suffix=classname_suffix)
    elif is_namedarraytuple(ntc):
        return type(ntc)
    elif is_namedarraytuple_class(ntc):
        return ntc
    else:
        raise TypeError("Input must be namedtuple or namedarraytuple instance"
            f" or class, got {type(ntc)}.")


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
