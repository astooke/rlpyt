
Base Classes and Interfaces
===========================

This page describes the base classes for three main components: algorithm, agent, and environment.  These are the most likely to need modification for a new project.  Intended interfaces to the infrastructure code (i.e. runner and sampler) are specified here.  More details on specific instances of these components appear in following pages.

Commonly, these classes will simply store their keyword arguments when instantiated, and actual initialization occurs in methods to be called later by the runner or sampler.  


Algorithms
----------

.. autoclass:: rlpyt.algos.base.RlAlgorithm
    :members: initialize, async_initialize, optim_initialize, optimize_agent, optim_state_dict, load_optim_state_dict


Environments
------------

Environments are expected to input/output numpy arrays.

.. autoclass:: rlpyt.envs.base.Env
    :members: step, reset, action_space, observation_space


Agents
------

Agents are expected to input/output torch tensors.

.. autoclass:: rlpyt.agents.base.BaseAgent
    :members: __init__, __call__, initialize, make_env_to_model_kwargs, to_device, data_parallel, async_cpu, step, state_dict, load_state_dict, train_mode, sample_mode, eval_mode, sync_shared_memory, send_shared_memory, recv_shared_memory

Recurrent Agents
^^^^^^^^^^^^^^^^

.. autoclass:: rlpyt.agents.base.RecurrentAgentMixin
    :members: reset, reset_one, advance_rnn_state, train_mode, sample_mode, eval_mode

.. autoclass:: rlpyt.agents.base.AlternatingRecurrentAgentMixin
