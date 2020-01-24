
Samplers
========

Several sampler classes are implemented for different parallelization schemes, with multiple environment instances running on CPU resources and agent forward passes happening on either CPU or GPU.  The implemented samplers execute a fixed number of time-steps at each call to ``obtain_samples()``, which returns a batch of data with leading dimensions ``[batch_T, batch_B]``.

Something about choosing which sampler based on parallel needs/availability, and different for each case, but try them out.

.. autoclass:: rlpyt.samplers.base.BaseSampler
    :members: initialize, obtain_samples, evaluate_agent

Serial Sampler
--------------

.. autoclass:: rlpyt.samplers.serial.sampler.SerialSampler
    :members: initialize, obtain_samples, evaluate_agent
    :show-inheritance:

Parallel Samplers
-----------------

.. autoclass:: rlpyt.samplers.parallel.base.ParallelSamplerBase
    :members: initialize, obtain_samples, evaluate_agent
    :show-inheritance:

CPU-Agent
^^^^^^^^^

.. autoclass:: rlpyt.samplers.parallel.cpu.sampler.CpuSampler
    :members: obtain_samples, evaluate_agent

GPU-Agent
^^^^^^^^^

.. autoclass:: rlpyt.samplers.parallel.gpu.sampler.GpuSamplerBase
    :members: obtain_samples, evaluate_agent, _agent_init
    :show-inheritance:

.. autoclass:: rlpyt.samplers.parallel.gpu.action_server.ActionServer
    :members: serve_actions, serve_actions_evaluation

.. autoclass:: rlpyt.samplers.parallel.gpu.sampler.GpuSampler
    :show-inheritance:

GPU-Agent, Alternating Workers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlpyt.samplers.parallel.gpu.alternating_sampler.AlternatingSamplerBase
    :members: initialize

.. autoclass:: rlpyt.samplers.parallel.gpu.action_server.AlternatingActionServer

.. autoclass:: rlpyt.samplers.parallel.gpu.action_server.NoOverlapAlternatingActionServer

.. autoclass:: rlpyt.samplers.parallel.gpu.alternating_sampler.AlternatingSampler
    :show-inheritance:

.. autoclass:: rlpyt.samplers.parallel.gpu.alternating_sampler.NoOverlapAlternatingSampler
    :show-inheritance:


Parallel Sampler Worker
-----------------------

The same function is used as the target for forking worker processes in all parallel samplers.

.. autofunction:: rlpyt.samplers.parallel.worker.sampling_process

.. autofunction:: rlpyt.samplers.parallel.worker.initialize_worker

