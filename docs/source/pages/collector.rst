
Collectors
==========

Collectors run the environment-agent interaction loop and record sampled data to the batch buffer.  The serial sampler runs one collector, and in parallel samplers, each worker process runs one collector.  Different collectors are needed for CPU-agent vs GPU-agent samplers.

In general, collectors will execute a for loop over time steps, and and inner for loop over environments, and step each environment one at a time.  At every step, all information (e.g. `observation`, `env_info`, etc.) is recorded to its place in the pre-allocated batch buffer.  All information is also fed to the trajectory-info object for each environment, for tracking trajectory-wise measures.  

Evaluation collectors only record trajectory-wise results.

Training Collectors
-------------------

Base Components
^^^^^^^^^^^^^^^

.. autoclass:: rlpyt.samplers.collectors.BaseCollector
    :members: start_envs, start_agent, collect_batch, reset_if_needed

.. autoclass:: rlpyt.samplers.collectors.DecorrelatingStartCollector
    :members: start_envs
    :show-inheritance:

CPU-Agent Collectors
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlpyt.samplers.parallel.cpu.collectors.CpuResetCollector
    :show-inheritance:

.. autoclass:: rlpyt.samplers.parallel.cpu.collectors.CpuWaitResetCollector
    :show-inheritance:

GPU-Agent Collectors
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rlpyt.samplers.parallel.gpu.collectors.GpuResetCollector
    :show-inheritance:

.. autoclass:: rlpyt.samplers.parallel.gpu.collectors.GpuWaitResetCollector
    :show-inheritance:

Evaluation Collectors
---------------------

.. autoclass:: rlpyt.samplers.collectors.BaseEvalCollector
    :members: collect_evaluation

.. autoclass:: rlpyt.samplers.parallel.cpu.collectors.CpuEvalCollector
    :show-inheritance:

.. autoclass:: rlpyt.samplers.parallel.gpu.collectors.GpuEvalCollector
    :show-inheritance: