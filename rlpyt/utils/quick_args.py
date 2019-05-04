
from inspect import getfullargspec


def save_args(values, underscore=False, overwrite=False):
    """
    Assign all args and kwargs to instance attributes.  To maintain
    precedence of args provided to subclasses, call this in the subclass before
    super().__init__() if save_args() also appears in base class.
    """
    prefix = "_" if underscore else ""
    self = values['self']
    args = list()
    for Cls in type(self).mro():  # class inheritances
        if '__init__' in vars(Cls):
            args += getfullargspec(Cls.__init__).args[1:]
    for arg in args:
        attr = prefix + arg
        if arg in values and (not hasattr(self, attr) or overwrite):
            setattr(self, attr, values[arg])
