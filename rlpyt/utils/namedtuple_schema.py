
# Classes for creating objects which closely follow the interfaces for
# namedtuple and namedarraytuple types and instances, except without
# defining a new class for each type.  (May be easier to use with regards
# to pickling or dynamically creating types, by avoiding module-level
# definitions.)


class NamedTupleSchema:
    """Instances of this class act like a type returned by namedtuple()."""

    def __init__(self, typename, fields):
        self._typename = typename
        self._fields = fields

    @property
    def typename(self):
        return self._typename

    @property
    def fields(self):
        return self._fields

    def __call__(self, *values, **kwargs):
        """Allows instances of `NamedTupleSchema` to act like `namedtuple`
        constructors."""
        if kwargs:
            try:
                values += tuple(map(kwargs.pop, self._fields[len(values):]))
            except KeyError as e:
                raise TypeError(f"Missing argument: '{e.args[0]}'")
            if kwargs:
                dup = [f for f in self._fields[:len(values)] if f in kwargs]
                if dup:
                    msg = (f"Got multiple values for argument(s): "
                           f"{str(dup)[1:-1]}")
                else:
                    msg = (f"Got unexpected field name(s): "
                           f"{str(list(kwargs))[1:-1]}")
                raise ValueError(msg)
        return self._make(values)

    def _make(self, values):
        return NamedTuple(self._typename, self._fields, values)


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
        if len(fields) != len(values):
            if len(fields) < len(values):
                msg = (f"__new__() '{typename}' requires {len(fields)}"
                       f" positional arguments but {len(values)} were given")
            else:
                num = len(fields) - len(values)
                msg = (f"__new__() '{typename}' missing {num} required "
                       f"positional arguments: {str(fields[-num:])[1:-1]}")
            raise TypeError(msg)
        result = tuple.__new__(cls, values)
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
        if hasattr(self, name):
            msg = "can't set attribute"
        else:
            msg = f"'{self._typename}' object has no attribute '{name}'"
        raise AttributeError(msg)

    def _make(self, values):
        """Make a new object of same typename and fields from a sequence or
        iterable."""
        return self.__new__(self.__class__, self._typename, self._fields,
                            values)

    def _replace(self, **kwargs):
        """Return a new object of same typename and fields, replacing specified
        fields with new values."""
        result = self._make(map(kwargs.pop, self._fields, self))
        if kwargs:
            raise ValueError(f"Got unexpected field names: "
                             f"{str(list(kwargs))[1:-1]}")
        return result

    def _asdict(self):
        """Return a new dictionary which maps field names to their values."""
        return dict(zip(self._fields, self))

    def __getnewargs__(self):
        """Returns typename, fields, and values as plain tuple. Used by copy
        and pickle."""
        return self._typename, self._fields, tuple(self)

    def __repr__(self):
        """Return a nicely formatted representation string showing the
        typename."""
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

    def _make(self, values):
        return NamedArrayTuple(self._typename, self._fields, values)


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
