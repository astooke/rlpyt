
Named Array Tuples
==================

.. autofunction:: rlpyt.utils.collections.namedarraytuple

.. autoclass:: rlpyt.utils.collections.DocExampleNat
    :members: __getitem__, __setitem__, __contains__, get, items

.. autofunction:: rlpyt.utils.collections.is_namedtuple_class

.. autofunction:: rlpyt.utils.collections.is_namedarraytuple_class

.. autofunction:: rlpyt.utils.collections.is_namedtuple

.. autofunction:: rlpyt.utils.collections.is_namedarraytuple

.. autofunction:: rlpyt.utils.collections.namedarraytuple_like

Alternative Implementation
--------------------------

Classes for creating objects which closely follow the interfaces for namedtuple and namedarraytuple types and instances, except without defining a new class for each type.  (May be easier to use with regards to pickling under spawn, or dynamically creating types, by avoiding module-level definitions.

.. autoclass:: rlpyt.utils.collections.NamedTupleSchema
    :members: __call__, _make

.. autoclass:: rlpyt.utils.collections.NamedTuple
    :members: __getattr__, _make, _replace, _asdict
    :show-inheritance:

.. autoclass:: rlpyt.utils.collections.NamedArrayTupleSchema
    :show-inheritance:

.. autoclass:: rlpyt.utils.collections.NamedArrayTuple
    :show-inheritance:

.. autofunction:: rlpyt.utils.collections.NamedArrayTupleSchema_like