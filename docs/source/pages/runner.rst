
Runners
=======

The runner orchstrates all the other components to run the training loop.  It serves as a single-point-of-entry into multi-GPU code.  During startup it initializes the sampler, algorithm, and agent, with care taken to do this in a specific order if
needed for multi-GPU.

The ``get_itr_snapshot()`` and ``save_itr_snapshot()`` methods can be used to save data or values during training such as the agent's parameters (but see the logger (link)).



`MinibatchRl`
-------------
Basic RL training loop which records performance using the "online" training samples.


`MinibatchRlEval`
-------------------
RL training loop which record performance using "offline" evaluation samples
(commonly used to reduce the exploratory behavior of the agent during
evaluation).

`SyncRl` & `SyncRlEval`
---------------------------
Synchronous multi-process RL.  Launches a separate instance of the entire runner (sampler, algo, agent) in each process, and they run independently except at synchronization points in the algorithm.  The main synchronization point will be in the gradient computation during the agent update (see agent/DDP), but ``torch.distributed`` can be used to explicitly communicate other values as needed (e.g. batch statistics).  Can run multi-GPU (via `nccl`) or multi-CPU (via `gloo`, like MPI).  In `SyncRl`, training trajectories from the workers are fed back to the main process and included in diagnostics logging; in `SyncRlEval`, only the master process runs evaluation trajectories.

.. note::
    The sampler and algorithm batch sizes will **not** be automatically adjusted according to GPU count.  For example, if an algorithm is initialized with ``batch_size=32``, and 2 GPUs are used, each process will use batch size 32, resulting in a total effective batch size of 64.

`AsyncRl` & `AsyncRlEval`
-------------------------
Asynchronous sampling-optimization in separate processes.  Each can be parallelized to their own extent, inclding with multiple GPUs each.  A double-buffer is created to hold sample batches in shared memory, and yet other processes are launched for the sole purpose of copying those into the main algorithm replay buffer (also on shared memory).  The master process will be the main optimization process.

.. note::
    The memory copier process uses the algorithm's ``sampler_to_buffer()`` method for selecting the desired components of the sample batch to store in the long-term buffer, but because it's running in a separate process, it will not have direct access to any changes to the algorithm's attributes performed in the master process.

The sampler will run at full speed, and the optimizer will throttle itself not to exceed the ``replay_ratio`` set point.  For example, if the replay ratio is 1, the sampler is generating 1,000 samples per second, and the optimizer uses batch size 100, then the optimizer will not exceed 10 updates per second.  However, say the optimizer is only able to perform 5 updates per second, then both sampler and optimizer will run at full speed, but the actual replay ratio achieved will only be 0.5.  This can significantly impact algorithm performance, so this value is recorded in the logs.  Each RL problem may require a different balance of sampler-optimizer resources.
