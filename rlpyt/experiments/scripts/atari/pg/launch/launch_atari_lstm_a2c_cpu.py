
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/atari/pg/train/atari_lstm_a2c_cpu.py"
default_config_key = "0"
affinity_code = encode_affinity(
    n_cpu_cores=6,
    n_gpu=2,
    hyperthread_offset=8,
    n_socket=1,
    # cpu_per_run=2,
)
runs_per_setting = 1
experiment_title = "first_lstm_test"
variant_levels = list()

learning_rate = [1e-4, 7e-4]
batch_T = [20, 20]
values = list(zip(learning_rate, batch_T))
names = ["test_{}lr_{}T".format(*v) for v in values]
keys = [("algo", "learning_rate"), ("sampler", "batch_T")]
variant_levels.append(VariantLevel(keys, values, names))


games = ["pong"]
values = list(zip(games))
names = ["{}".format(*v) for v in values]
keys = [("env", "game")]
variant_levels.append(VariantLevel(keys, values, names))

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
