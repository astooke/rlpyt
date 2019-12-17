
Runners
=======

.. autoclass:: rlpyt.runners.base.BaseRunner
    :members: train

All of the existing runners implement loops which collect minibatches of samples and provide them to the algorithm.  The distinguishing features of the following classes are: a) online vs offline performance logging, b) single- vs multi-GPU training, and c) synchronous vs asynchronous operation of sampling and training.  Most RL workflows should be able to use the desired class without modification.


Single-GPU Runners
------------------
.. autoclass:: rlpyt.runners.minibatch_rl.MinibatchRlBase
    :members: startup, get_traj_info_kwargs, get_n_itr, get_itr_snapshot, save_itr_snapshot, store_diagnostics, log_diagnostics, _log_infos
    :show-inheritance:

.. autoclass:: rlpyt.runners.minibatch_rl.MinibatchRl
    :members: __init__, train
    :show-inheritance:

.. autoclass:: rlpyt.runners.minibatch_rl.MinibatchRlEval
    :members: train, evaluate_agent
    :show-inheritance:


Multi-GPU Runners
-----------------
.. autoclass:: rlpyt.runners.sync_rl.SyncRlMixin
    :members: launch_workers

.. autoclass:: rlpyt.runners.sync_rl.SyncRl
    :show-inheritance:

.. autoclass:: rlpyt.runners.sync_rl.SyncRlEval
    :show-inheritance:


Asynchronous Runners
--------------------

.. autoclass:: rlpyt.runners.async_rl.AsyncRlBase
    :members: startup, optim_startup, train, build_ctrl, launch_optimizer_workers, launch_memcpy
    :show-inheritance:

.. autoclass:: rlpyt.runners.async_rl.AsyncRl
    :show-inheritance:

.. autoclass:: rlpyt.runners.async_rl.AsyncRlEval
    :show-inheritance:


Asynchronous Worker Processes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: rlpyt.runners.async_rl.run_async_sampler

.. autofunction:: rlpyt.runners.async_rl.run_async_sampler_eval

.. autofunction:: rlpyt.runners.async_rl.memory_copier
