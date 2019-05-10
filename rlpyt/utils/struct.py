

class Struct(dict):
    """
    Behaves like a dictionary but additionally has attribute-style access
    for both read and write.
    e.g. x["key"] and x.key are the same,
    e.g. can iterate using:  for k, v in x.items().

    Can sublcass for specific data classes; must call Struct's __init__().
    """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

    def copy(self):
        """
        Provides a "deep copy" of all unbroken chains of types (Struct, dict,
        list), but shallow copies otherwise, so no data is actually copied
        (e.g. numpy arrays are NOT copied).
        """
        new_dict = dict()
        for k, v in self.__dict__.items():
            new_dict[k] = _struct_copy(v)
        return Struct(**new_dict)


def _struct_copy(obj):
    if isinstance(obj, (dict, list, Struct)):
        obj = obj.copy()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list, Struct)):
                obj[k] = _struct_copy(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list, Struct)):
                obj[i] = _struct_copy(v)
    return obj
