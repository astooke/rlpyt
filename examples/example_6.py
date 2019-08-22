
"""
Launches multiple experiment runs and organizes them on the local
compute resource.
Processor (CPU and GPU) affinities are all specified, to keep each
experiment on its own hardware without interference.  Can queue up more
experiments than fit on the machine, and they will run in order over time.  

To understand rules and settings for affinities, try using 
affinity = affinity.make_affinity(..)
OR
code = affinity.encode_affinity(..)
slot_code = affinity.prepend_run_slot(code, slot)
affinity = affinity.affinity_from_code(slot_code)
with many different inputs to encode, and see what comes out.

The results will be logged with a folder structure according to the
variant levels constructed here.

"""

from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

# Either manually set the resources for the experiment:
affinity_code = encode_affinity(
    n_cpu_core=2,
    n_gpu=0,
    # hyperthread_offset=8,  # if auto-detect doesn't work, number of CPU cores
    # n_socket=1,  # if auto-detect doesn't work, can force (or force to 1)
    cpu_per_run=1,
    set_affinity=True,  # it can help to restrict workers to individual CPUs
)
# Or try an automatic one, but results may vary:
# affinity_code = quick_affinity_code(n_parallel=None, use_gpu=True)

runs_per_setting = 2
experiment_title = "example_6"
variant_levels = list()

# Within a variant level, list each combination explicitly.
learning_rate = [7e-4, 1e-3]
batch_B = [16, 32]
values = list(zip(learning_rate, batch_B))
dir_names = ["example6_{}lr_{}B".format(*v) for v in values]
keys = [("algo", "learning_rate"), ("sampler", "batch_B")]
variant_levels.append(VariantLevel(keys, values, dir_names))


games = ["pong", "seaquest"]
values = list(zip(games))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "game")]
variant_levels.append(VariantLevel(keys, values, dir_names))

# Between variant levels, make all combinations.
variants, log_dirs = make_variants(*variant_levels)

run_experiments(
    script="examples/example_6a.py",
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
)
