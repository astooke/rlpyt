
import sys
import copy

from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

args = sys.argv[1:]
assert len(args) == 2
my_computer = int(args[0])
num_computers = int(args[1])

print(f"MY_COMPUTER: {my_computer},  NUM_COMPUTERS: {num_computers}")

script = "rlpyt/ul/experiments/rl_from_ul/scripts/atari/train/atari_ppo_from_ul_serial.py"

affinity_code = quick_affinity_code(contexts_per_gpu=3)
runs_per_setting = 3
experiment_title = "ppo_from_inv_first_1"

variant_levels_1 = list()
variant_levels_2 = list()
# variant_levels_3 = list()

learning_rates = [1e-3]
lr_schedules = ["cosine"]
lr_warmups = [1e3]
values = list(zip(learning_rates, lr_schedules, lr_warmups))
dir_names = ["{}lr_{}sched_{}wrmp".format(*v) for v in values]
keys = [("pretrain", "learning_rate"), ("pretrain", "learning_rate_anneal"),
    ("pretrain", "learning_rate_warmup")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

n_updates = [20e3, 100e3]
values = list(zip(n_updates))
dir_names = ["{}updates".format(*v) for v in values]
keys = [("pretrain", "n_updates")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

random_shift_probs = [0., 1.]
values = list(zip(random_shift_probs))
dir_names = ["{}rsprob".format(*v) for v in values]
keys = [("pretrain", "random_shift_prob")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

n_steps_predict = [1, 3]
values = list(zip(n_steps_predict))
dir_names = ["{}nstepspredict".format(*v) for v in values]
keys = [("pretrain", "n_steps_predict")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))


# games = ["pong", "qbert", "seaquest", "space_invaders",
#     "alien", "breakout", "frostbite", "gravitar"]
games = ["breakout", "gravitar", "qbert", "space_invaders"]
values = list(zip(games))
dir_names = games
keys = [("env", "game")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))
# variant_levels_2.append(VariantLevel(keys, values, dir_names))
# variant_levels_3.append(VariantLevel(keys, values, dir_names))


##################################################
# RL CONFIG (mostly)

n_steps = [25e6]
pretrain_algos = ["Inverse"]
replays = ["20200608/15M_VecEps_B78"]
model_dirs = ["/data/adam/ul4rl/models/20200824/atari_ul_inverse_first_1/"]
values = list(zip(
    n_steps,
    pretrain_algos,
    replays,
    model_dirs,
))
dir_names = ["RlFromUl"]  # TRAIN SCRIPT SPLITS OFF THIS
keys = [
    ("runner", "n_steps"),
    ("pretrain", "algo"),
    ("pretrain", "replay"),
    ("pretrain", "model_dir"),
]
variant_levels_1.append(VariantLevel(keys, values, dir_names))
# variant_levels_2.append(VariantLevel(keys, values, dir_names))


stop_grads = ["conv"]
load_fc1s = [False]
hidden_sizes = [512]  # I guess to be fair?  there is also fc1=512
values = list(zip(stop_grads, load_fc1s, hidden_sizes))
dir_names = ["{}_stpgrd_{}loadfc1_{}hs".format(*v) for v in values]
keys = [("model", "stop_grad"), ("agent", "load_fc1"),
    ("model", "hidden_sizes")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))
# variant_levels_2.append(VariantLevel(keys, values, dir_names))

normalize_convs = [False, True]
values = list(zip(normalize_convs))
dir_names = ["{}normconv".format(*v) for v in values]
keys = [("model", "normalize_conv_out")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

variants_1, log_dirs_1 = make_variants(*variant_levels_1)
# variants_2, log_dirs_2 = make_variants(*variant_levels_2)

variants = variants_1  # + variants_2
log_dirs = log_dirs_1  # + log_dirs_2

num_variants = len(variants)
variants_per = num_variants // num_computers

my_start = my_computer * variants_per
if my_computer == num_computers - 1:
    my_end = num_variants
else:
    my_end = (my_computer + 1) * variants_per
my_variants = variants[my_start:my_end]
my_log_dirs = log_dirs[my_start:my_end]

default_config_key = "ppo_16env"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=my_variants,
    log_dirs=my_log_dirs,
    common_args=(default_config_key, experiment_title),
)

