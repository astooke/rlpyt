
import sys
from collections import namedtuple, OrderedDict
from inspect import Signature as Sig, Parameter as Param
import string

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
    indexing (__getitem__) behavior (e.g. numpy arrays or torch tensors).

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


DocExampleNat = namedarraytuple("DocExampleNat", ["field_1", "field_2"])


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
    namedarraytuple instance or class, the same class is returned directly.
    Input can also be from the new Schema format, instances of the four:
    Named[Array]Tuple[Schema]."""
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
    elif isinstance(ntc, (NamedTupleSchema, NamedTuple, NamedArrayTupleSchema,
            NamedArrayTuple)):
        return namedarraytuple(ntc._typename, ntc._fields,
            classname_suffix=classname_suffix)
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


############################################################################
# Classes for creating objects which closely follow the interfaces for
# namedtuple and namedarraytuple types and instances, except without defining
# a new class for each type.  (May be easier to use with regards to pickling
# under spawn, or dynamically creating types, by avoiding module-level
# definitions.)
############################################################################


class NamedTupleSchema:
    """Instances of this class act like a type returned by namedtuple()."""

    def __init__(self, typename, fields):
        if not isinstance(typename, str):
            raise TypeError(f"type name must be string, not {type(typename)}")

        if isinstance(fields, str):
            spaces = any([whitespace in fields for whitespace in string.whitespace])
            commas = "," in fields
            if spaces and commas:
                raise ValueError(f"Single string fields={fields} cannot have both spaces and commas.")
            elif spaces:
                fields = fields.split()
            elif commas:
                fields = fields.split(",")
            else:
                # If there are neither spaces nor commas, then there is only one field.
                fields = (fields,)
        fields = tuple(fields)

        for field in fields:
            if not isinstance(field, str):
                raise ValueError(f"field names must be strings: {field}")
            if field.startswith("_"):
                raise ValueError(f"field names cannot start with an "
                                 f"underscore: {field}")
            if field in ("index", "count"):
                raise ValueError(f"can't name field 'index' or 'count'")
        self.__dict__["_typename"] = typename
        self.__dict__["_fields"] = fields
        self.__dict__["_signature"] = Sig(Param(field,
            Param.POSITIONAL_OR_KEYWORD) for field in fields)

    def __call__(self, *args, **kwargs):
        """Allows instances to act like `namedtuple` constructors."""
        args = self._signature.bind(*args, **kwargs).args  # Mimic signature.
        return self._make(args)

    def _make(self, iterable):
        """Allows instances to act like `namedtuple` constructors."""
        return NamedTuple(self._typename, self._fields, iterable)

    def __setattr__(self, name, value):
        """Make the type-like object immutable."""
        raise TypeError(f"can't set attributes of '{type(self).__name__}' "
                        "instance")

    def __repr__(self):
        return f"{type(self).__name__}({self._typename!r}, {self._fields!r})"


class NamedTuple(tuple):
    """
    Instances of this class act like instances of namedtuple types, but this
    same class is used for all namedtuple-like types created.  Unlike true
    namedtuples, this mock avoids defining a new class for each configuration
    of typename and field names.  Methods from namedtuple source are copied
    here.


    Implementation differences from `namedtuple`:

    * The individual fields don't show up in `dir(obj)`, but they do still
      show up as `hasattr(obj, field) => True`, because of `__getattr__()`.
    * These objects have a `__dict__` (by ommitting `__slots__ = ()`),
      intended to hold only the typename and list of field names, which are
      now instance attributes instead of class attributes.
    * Since `property(itemgetter(i))` only works on classes, `__getattr__()`
      is modified instead to look for field names.
    * Attempts to enforce call signatures are included, might not exactly
      match.
    """

    def __new__(cls, typename, fields, values):
        result = tuple.__new__(cls, values)
        if len(fields) != len(result):
            raise ValueError(f"Expected {len(fields)} arguments, got "
                             f"{len(result)}")
        result.__dict__["_typename"] = typename
        result.__dict__["_fields"] = fields
        return result

    def __getattr__(self, name):
        """Look in `_fields` when `name` is not in `dir(self)`."""
        try:
            return tuple.__getitem__(self, self._fields.index(name))
        except ValueError:
            raise AttributeError(f"'{self._typename}' object has no attribute "
                                 f"'{name}'")

    def __setattr__(self, name, value):
        """Make the object immutable, like a tuple."""
        raise AttributeError(f"can't set attributes of "
                             f"'{type(self).__name__}' instance")

    def _make(self, iterable):
        """Make a new object of same typename and fields from a sequence or
        iterable."""
        return type(self)(self._typename, self._fields, iterable)

    def _replace(self, **kwargs):
        """Return a new object of same typename and fields, replacing specified
        fields with new values."""
        result = self._make(map(kwargs.pop, self._fields, self))
        if kwargs:
            raise ValueError(f"Got unexpected field names: "
                             f"{str(list(kwargs))[1:-1]}")
        return result

    def _asdict(self):
        """Return an ordered dictionary mapping field names to their values."""
        return OrderedDict(zip(self._fields, self))

    def __getnewargs__(self):
        """Returns typename, fields, and values as plain tuple. Used by copy
        and pickle."""
        return self._typename, self._fields, tuple(self)

    def __repr__(self):
        """Return a nicely formatted string showing the typename."""
        return self._typename + '(' + ', '.join(f'{name}={value}'
            for name, value in zip(self._fields, self)) + ')'


class NamedArrayTupleSchema(NamedTupleSchema):
    """Instances of this class act like a type returned by rlpyt's
    namedarraytuple()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in self._fields:
            if name in RESERVED_NAMES:
                raise ValueError(f"Disallowed field name: '{name}'")

    def _make(self, iterable):
        return NamedArrayTuple(self._typename, self._fields, iterable)


class NamedArrayTuple(NamedTuple):

    def __getitem__(self, loc):
        """Return a new object of the same typename and fields containing the
        selected index or slice from each value."""
        try:
            return self._make(None if s is None else s[loc] for s in self)
        except IndexError as e:
            for j, s in enumerate(self):
                if s is None:
                    continue
                try:
                    _ = s[loc]
                except IndexError:
                    raise Exception(f"Occured in '{self._typename}' at field "
                                    f"'{self._fields[j]}'.") from e

    def __setitem__(self, loc, value):
        """
        If input value is the same named[array]tuple type, iterate through its
        fields and assign values into selected index or slice of corresponding
        value.  Else, assign whole of value to selected index or slice of
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
        """Checks presence of field name (unlike tuple; like dict)."""
        return key in self._fields

    def get(self, index):
        """Retrieve value as if indexing into regular tuple."""
        return tuple.__getitem__(self, index)

    def items(self):
        """Iterate ordered (field_name, value) pairs (like OrderedDict)."""
        for k, v in zip(self._fields, self):
            yield k, v


def NamedArrayTupleSchema_like(example):
    """Returns a NamedArrayTupleSchema instance  with the same name and fields
    as input, which can be a class or instance of namedtuple or
    namedarraytuple, or an instance of NamedTupleSchema, NamedTuple,
    NamedArrayTupleSchema, or NamedArrayTuple."""
    if isinstance(example, NamedArrayTupleSchema):
        return example
    elif isinstance(example, (NamedArrayTuple, NamedTuple, NamedTupleSchema)):
        return NamedArrayTupleSchema(example._typename, example._fields)
    elif is_namedtuple(example) or is_namedarraytuple(example):
        return NamedArrayTupleSchema(type(example).__name__, example._fields)
    elif is_namedtuple_class(example) or is_namedarraytuple_class(example):
        return NamedArrayTupleSchema(example.__name__, example._fields)
    else:
        raise TypeError("Input must be namedtuple or namedarraytuple instance"
            f" or class, or Named[Array]Tuple[Schema] instance, got "
            f"{type(example)}.")
