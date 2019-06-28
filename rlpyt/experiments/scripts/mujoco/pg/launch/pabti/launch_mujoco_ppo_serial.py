
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/mujoco/pg/train/mujoco_ff_ppo_serial.py"
affinity_code = encode_affinity(
    n_cpu_cores=16,
    n_gpu=8,
    contexts_per_gpu=2,
    hyperthread_offset=24,
    n_socket=2,
    # cpu_per_run=2,
)
runs_per_setting = 4
default_config_key = "ppo_1M_serial"
experiment_title = "ppo_mujoco_serial"
variant_levels = list()

env_ids = ["Hopper-v3", "Swimmer-v3", "HalfCheetah-v3",
    "Walker2d-v3", "Ant-v3", "Humanoid-v3"]
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
