
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/atari/dqn/train/atari_dqn_gpu.py"
affinity_code = encode_affinity(
    n_cpu_cores=16,
    n_gpu=4,
    hyperthread_offset=20,
    n_socket=2,
)
runs_per_setting = 1
experiment_title = "atari_dqn_pong_got"
variant_levels = list()


learning_rates = [5e-5, 1e-4, 2.5e-4, 5e-4]
values = list(zip(learning_rates))
dir_names = ["{}".format(*v) for v in values]
keys = [("algo", "learning_rate")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

default_config_key = "short_run"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
