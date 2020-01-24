
Asynchronous Samplers
=====================

Separate sampler classes are needed for asynchronous sampling-optimization mode, and they closely match the options for the other samplers.  In asynchronous mode, the sampler will run in a separate process forked from the main (training) process.  Parallel asynchronous samplers will fork additional processes.

Base Components
---------------

.. autoclass:: rlpyt.samplers.async_.base.AsyncSamplerMixin
    :members: async_initialize

.. autoclass:: rlpyt.samplers.async_.base.AsyncParallelSamplerMixin
    :members: obtain_samples
    :show-inheritance:


Serial
------

.. autoclass:: rlpyt.samplers.async_.serial_sampler.AsyncSerialSampler
    :members: initialize, obtain_samples, evaluate_agent
    :show-inheritance:

CPU-Agent
---------
.. autoclass:: rlpyt.samplers.async_.cpu_sampler.AsyncCpuSampler
    :members: initialize, obtain_samples, evaluate_agent
    :show-inheritance:

GPU-Agent
---------

Main Class
^^^^^^^^^^
.. autoclass:: rlpyt.samplers.async_.gpu_sampler.AsyncGpuSampler
    :show-inheritance:

Component Definitions
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlpyt.samplers.async_.gpu_sampler.AsyncGpuSamplerBase
    :members: initialize, action_server_process
    :show-inheritance:

.. autoclass:: rlpyt.samplers.async_.action_server.AsyncActionServer
    :members: serve_actions_evaluation
    :show-inheritance:


GPU-Agent, Alternating Workers
------------------------------

Main Classes
^^^^^^^^^^^^

.. autoclass:: rlpyt.samplers.async_.alternating_sampler.AsyncAlternatingSampler
    :show-inheritance:

.. autoclass:: rlpyt.samplers.async_.alternating_sampler.AsyncNoOverlapAlternatingSampler
    :show-inheritance:


Component Definitions
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlpyt.samplers.async_.alternating_sampler.AsyncAlternatingSamplerBase
    :show-inheritance:

.. autoclass:: rlpyt.samplers.async_.action_server.AsyncAlternatingActionServer
    :members: serve_actions_evaluation
    :show-inheritance:

.. autoclass:: rlpyt.samplers.async_.action_server.AsyncNoOverlapAlternatingActionServer
    :members: serve_actions_evaluation
    :show-inheritance:

