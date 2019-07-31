
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/mujoco/qpg/train/mujoco_td3_async_cpu.py"
affinity_code = encode_affinity(
    n_cpu_core=16,
    n_gpu=4,
    # contexts_per_gpu=2,
    async_sample=True,
    # hyperthread_offset=2,
    # n_socket=1,
    # cpu_per_run=1,
)
runs_per_setting = 2
default_config_key = "async_cpu"
experiment_title = "td3_mujoco_async"
variant_levels = list()

env_ids = ["Hopper-v3", "HalfCheetah-v3"]  # , "Swimmer-v3"]
values = list(zip(env_ids))
dir_names = ["env_{}".format(*v) for v in values]
keys = [("env", "id")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
