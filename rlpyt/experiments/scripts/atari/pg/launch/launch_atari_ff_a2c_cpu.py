
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/atari/pg/train/atari_ff_a2c_cpu.py"
default_config_key = "0"
affinity_code = encode_affinity(
    n_cpu_cores=4,
    n_gpu=2,
    hyperthread_offset=8,
    n_socket=1,
    # cpu_per_run=4,
)
runs_per_setting = 2
experiment_title = "ff_retest_GPU_opt"
variant_levels = list()

learning_rate = [7e-4]
batch_B = [32]
values = list(zip(learning_rate, batch_B))
dir_names = ["test_{}lr_{}B".format(*v) for v in values]
keys = [("algo", "learning_rate"), ("sampler", "batch_B")]
variant_levels.append(VariantLevel(keys, values, dir_names))


games = ["pong", "seaquest"]
values = list(zip(games))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "game")]
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
