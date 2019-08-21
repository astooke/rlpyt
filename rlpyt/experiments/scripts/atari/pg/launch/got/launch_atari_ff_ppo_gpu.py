
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

affinity_code = encode_affinity(
    n_cpu_core=16,
    n_gpu=4,
    hyperthread_offset=20,
    n_socket=2
    # cpu_per_run=2,
)
runs_per_setting = 2

variant_levels = list()

games = ["pong", "seaquest", "qbert", "chopper_command"]
values = list(zip(games))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "game")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)


# script = "rlpyt/experiments/scripts/atari/pg/train/atari_ff_a2c_gpu.py"
# experiment_title = "atari_ff_a2c_basic"
# default_config_key = "0"

# run_experiments(
#     script=script,
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
#     common_args=(default_config_key,),
# )

# script = "rlpyt/experiments/scripts/atari/pg/train/atari_lstm_a2c_gpu.py"
# experiment_title = "atari_lstm_a2c_basic"
# default_config_key = "0"

# run_experiments(
#     script=script,
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
#     common_args=(default_config_key,),
# )


script = "rlpyt/experiments/scripts/atari/pg/train/atari_ff_ppo_gpu.py"
experiment_title = "atari_ff_ppo_basic"
default_config_key = "0"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
