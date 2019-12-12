
Base Classes and Interfaces
===========================

This page describes the base classes for several main components: runner, algorithm, agent, and environment, including how they are intended to interact.  More details on specific instances of these components appear in following pages.

Commonly, these classes will simply store their keyword arguments when instantiated, and actual initialization occurs in methods to be called later.


Runner
------

.. autoclass:: rlpyt.runners.base.BaseRunner
   :members: train




Algorithm
---------

.. autoclass:: rlpyt.algos.base.RlAlgorithm
    :members: initialize, async_initialize, optim_initialize, optimize_agent, optim_state_dict, load_optim_state_dict



Agent
-----
The agent class performs many functions, including: action-selection during sampling, returning policy-relevant values to use in training (e.g. action probabilities), storing recurrent state during sampling, managing model device, and performing model parameter communication between processes.  The agent is the interface between neural network and sampler and between neural network and algorithm.  Typically, each algorithm and environment combination will require its own agent functionality.

