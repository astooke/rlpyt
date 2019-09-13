
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/mujoco/pg/train/mujoco_ppo_serial.py"
affinity_code = encode_affinity(
    n_cpu_core=4,
    n_gpu=4,
    # contexts_per_gpu=1,
    # hyperthread_offset=24,
    # n_socket=2,
    # cpu_per_run=2,
)
runs_per_setting = 4
default_config_key = "ppo_1M_serial"
experiment_title = "ppo_mujoco_v3_serial_hc_tl"
# variant_levels_1M = list()
variant_levels_3M = list()

# n_steps = [1e6]
# values = list(zip(n_steps))
# dir_names = ["1M"]
# keys = [("runner", "n_steps")]
# variant_levels_1M.append(VariantLevel(keys, values, dir_names))

bootstrap_tls = [True]
values = list(zip(bootstrap_tls))
dir_names = ["bootstrap_timelimit"]
keys = [("algo", "bootstrap_timelimit")]
variant_levels_3M.append(VariantLevel(keys, values, dir_names))

n_steps = [3e6]
values = list(zip(n_steps))
dir_names = ["3M"]
keys = [("runner", "n_steps")]
variant_levels_3M.append(VariantLevel(keys, values, dir_names))


# env_ids = ["Hopper-v3", "Walker2d-v3"]
# values = list(zip(env_ids))
# dir_names = ["{}".format(*v) for v in values]
# keys = [("env", "id")]
# variant_levels_1M.append(VariantLevel(keys, values, dir_names))

# env_ids = ["Ant-v3", "HalfCheetah-v3"]
env_ids = ["HalfCheetah-v3"]
values = list(zip(env_ids))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "id")]
variant_levels_3M.append(VariantLevel(keys, values, dir_names))


# variants_1M, log_dirs_1M = make_variants(*variant_levels_1M)
variants_3M, log_dirs_3M = make_variants(*variant_levels_3M)
variants = variants_3M  # + variants_1M
log_dirs = log_dirs_3M  # + log_dirs_1M

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)

