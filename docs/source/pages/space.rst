
Spaces
======

Spaces are used to specify the interfaces from the environment to the agent (model); the describe the observations and actions.

.. autoclass:: rlpyt.spaces.base.Space
    :members: sample, null_value

.. autoclass:: rlpyt.spaces.int_box.IntBox
    :members: __init__, sample, n
    :show-inheritance:

.. autoclass:: rlpyt.spaces.float_box.FloatBox
    :members: __init__, sample
    :show-inheritance:

.. autoclass:: rlpyt.spaces.composite.Composite
    :members: __init__, sample, null_value, shape, names, spaces
    :show-inheritance:

.. autoclass:: rlpyt.spaces.gym_wrapper.GymSpaceWrapper
    :members: __init__, sample, null_value, convert, revert
