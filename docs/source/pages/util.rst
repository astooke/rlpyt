
Utilities
=========

Here are listed number of miscellaneous utilities used in rlpyt.

Array
-----

Miscellaneous functions for manipulating numpy arrays.

.. autofunction:: rlpyt.utils.array.select_at_indexes

.. autofunction:: rlpyt.utils.array.to_onehot

.. autofunction:: rlpyt.utils.array.valid_mean

.. autofunction:: rlpyt.utils.array.infer_leading_dims


Tensor
------

Miscellaneous functions for manipulating torch tensors.

.. autofunction:: rlpyt.utils.tensor.select_at_indexes

.. autofunction:: rlpyt.utils.tensor.to_onehot

.. autofunction:: rlpyt.utils.tensor.from_onehot

.. autofunction:: rlpyt.utils.tensor.valid_mean

.. autofunction:: rlpyt.utils.tensor.infer_leading_dims

.. autofunction:: rlpyt.utils.tensor.restore_leading_dims


Miscellaneous Array / Tensor
----------------------------

.. autofunction:: rlpyt.utils.misc.iterate_mb_idxs

.. autofunction:: rlpyt.utils.misc.zeros

.. autofunction:: rlpyt.utils.misc.empty

.. autofunction:: rlpyt.utils.misc.extract_sequences



Collections
-----------

(see Named Array Tuple page)

.. autoclass:: rlpyt.utils.collections.AttrDict
    :members: copy
    :show-inheritance:


Buffers
-------

.. autofunction:: rlpyt.utils.buffer.buffer_from_example

.. autofunction:: rlpyt.utils.buffer.build_array

.. autofunction:: rlpyt.utils.buffer.np_mp_array

.. autofunction:: rlpyt.utils.buffer.torchify_buffer

.. autofunction:: rlpyt.utils.buffer.numpify_buffer

.. autofunction:: rlpyt.utils.buffer.buffer_to

.. autofunction:: rlpyt.utils.buffer.buffer_method

.. autofunction:: rlpyt.utils.buffer.buffer_func

.. autofunction:: rlpyt.utils.buffer.get_leading_dims

Algorithms
----------

.. autofunction:: rlpyt.algos.utils.discount_return

.. autofunction:: rlpyt.algos.utils.generalized_advantage_estimation

.. autofunction:: rlpyt.algos.utils.discount_return_n_step

.. autofunction:: rlpyt.algos.utils.valid_from_done

.. autofunction:: rlpyt.algos.utils.discount_return_tl

.. autofunction:: rlpyt.algos.utils.generalized_advantage_estimation_tl

Synchronize
-----------

.. autoclass:: rlpyt.utils.synchronize.RWLock

.. autofunction:: rlpyt.utils.synchronize.drain_queue

.. autofunction:: rlpyt.utils.synchronize.find_port


Quick Arguments
---------------

.. autofunction:: rlpyt.utils.quick_args.save__init__args


Progress Bar
------------

.. autoclass:: rlpyt.utils.prog_bar.ProgBarCounter


Seed
----

.. autofunction:: rlpyt.utils.seed.set_seed

.. autofunction:: rlpyt.utils.seed.make_seed


