
# Classes for creating objects which closely follow the interfaces for
# namedtuple and namedarraytuple types and instances, except without
# defining a new class for each type.  (May be easier to use with regards
# to pickling or dynamically creating types, by avoiding module-level
# definitions.)


from rlpyt.utils.collections import (is_namedtuple, is_namedarraytuple,
    is_namedtuple_class, is_namedarraytuple_class)
from collections import OrderedDict
from inspect import Signature as Sig, Parameter as Param


class NamedTupleSchema:
    """Instances of this class act like a type returned by namedtuple()."""

    def __init__(self, typename, fields):
        if not isinstance(typename, str):
            raise TypeError(f"type name must be string, not {type(typename)}")
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
        args = self._signature.bind(*args, **kwargs).args
        return self._make(args)

    def _make(self, iterable):
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
    -The individual fields don't show up in dir(obj), but they do still show up
     as `hasattr(obj, field) => True`, because of __getattr__().
    -These objects have a __dict__ (by ommitting __slots__ = ()), intended to
     hold only the typename and list of field names, which are now instance
     attributes instead of class attributes.
    -Since property(itemgetter(i)) only works on classes, __getattr__() is
     modified instead to look for field names.
    -Attempts to enforce call signatures are included, might not exactly match.
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


RESERVED_NAMES = ("get", "items")


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
    namedarraytuple, or an instance of NamedTupleScheme, NamedTuple,
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
