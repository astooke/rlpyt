
Creating and Launching Experiments
==================================

Some utilities are included for creating and launching experiments comprised of multiple individual learning runs, e.g. for hyperparameter sweeps.  To date, these include functions for launching locally on a machine, so launching into the cloud may require different tooling.  Many experiments can be queued on a given hardware resource, and they will be cycled through to run in sequence (e.g. a desktop with 4 GPUs and each run getting exclusive use of 2 GPUs).

Launching
---------

.. autofunction:: rlpyt.utils.launching.exp_launcher.run_experiments

.. autofunction:: rlpyt.utils.launching.exp_launcher.launch_experiment

Variants
--------

Some simple tools are provided for creating hyperparameter value variants.

.. autoclass:: rlpyt.utils.launching.variant.VariantLevel

.. autofunction:: rlpyt.utils.launching.variant.make_variants

.. autofunction:: rlpyt.utils.launching.variant._cross_variants

.. autofunction:: rlpyt.utils.launching.variant.load_variant

.. autofunction:: rlpyt.utils.launching.variant.save_variant

.. autofunction:: rlpyt.utils.launching.variant.update_config


Affinity
--------

The hardware affinity is used for several purposes: 1) the experiment launcher uses it to determine how many concurrent experiments to run, 2) runners use it to determine GPU device selection, 3) parallel samplers use it to determine the number of worker processes, and 4) multi-GPU and asynchronous runners use it to determine the number of parallel processes.  The main intent of the implemented utilities is to take as input the total amount of hardware resources in the computer (CPU & GPU) and the amount of resources to be dedicated to each job, and then to divide resources evenly.  

.. admonition:: Example

    An 8-GPU, 40-CPU machine would have 5 CPU assigned to each GPU.  1 GPU per run would set up 8 concurrent experiments, with each sampler using the 5 CPU.  2 GPU per run with synchronous runner would set up 4 concurrent experiments.


.. autofunction:: rlpyt.utils.launching.affinity.encode_affinity

.. autofunction:: rlpyt.utils.launching.affinity.encode_affinity

.. autofunction:: rlpyt.utils.launching.affinity.make_affinity

.. autofunction:: rlpyt.utils.launching.affinity.affinity_from_code
