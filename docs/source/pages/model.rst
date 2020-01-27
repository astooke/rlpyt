
Model Components
================

This page documents the implemented neural network components.  These are intended as building blocks for the agent model, but not to be used as standalone models (should probably disambiguate the name from `model`).

Complete models which actually function as the agent model have additional functionality in the ``forward()`` method for handling of leading dimensions of inputs/outputs.  See ``infer_leading_dims()`` and ``restore_leading_dims`` until utilities, and see the documentation for each algorithm for associated complete models.

.. autoclass:: rlpyt.models.mlp.MlpModel
    :members: forward, output_size
    :show-inheritance:

.. autoclass:: rlpyt.models.conv2d.Conv2dModel
    :members: forward, conv_out_size
    :show-inheritance:

.. autoclass:: rlpyt.models.conv2d.Conv2dHeadModel
    :members: forward, output_size
    :show-inheritance:

Utilities
---------

.. autofunction:: rlpyt.models.utils.conv2d_output_shape

.. autoclass:: rlpyt.models.utils.ScaleGrad
    :members: forward, backward

.. autofunction:: rlpyt.models.utils.update_state_dict

.. autofunction:: rlpyt.models.utils.strip_ddp_state_dict
