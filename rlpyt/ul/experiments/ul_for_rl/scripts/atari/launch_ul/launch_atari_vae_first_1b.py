
import sys
import copy
import os.path as osp

from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

args = sys.argv[1:]
assert len(args) == 2 or len(args) == 0
if len(args) == 0:
    my_computer = 0
    num_computers = 1
elif len(args) == 2:
    my_computer = int(args[0])  # 0-indexed
    num_computers = int(args[1])

print(f"MY_COMPUTER: {my_computer},  NUM_COMPUTERS: {num_computers}")

script = "rlpyt/ul/experiments/ul_for_rl/scripts/atari/train_ul/atari_vae.py"

affinity_code = quick_affinity_code(contexts_per_gpu=2)
runs_per_setting = 1
experiment_title = "atari_ul_vae_first_1"
variant_levels = list()

learning_rates = [1e-3]
lr_schedules = ["cosine"]
lr_warmups = [1e3]
values = list(zip(learning_rates, lr_schedules, lr_warmups))
dir_names = ["{}lr_{}sched_{}wrmp".format(*v) for v in values]
keys = [("algo", "learning_rate"), ("algo", "learning_rate_anneal"),
    ("algo", "learning_rate_warmup")]
variant_levels.append(VariantLevel(keys, values, dir_names))

n_updates = [20e3, 100e3]
values = list(zip(n_updates))
dir_names = ["{}updates".format(*v) for v in values]
keys = [("runner", "n_updates")]
variant_levels.append(VariantLevel(keys, values, dir_names))

# n_steps_predict = [0, 1, 3]
# hidden_sizes = [None, 512, 512]
n_steps_predict = [3]
hidden_sizes = [512]
values = list(zip(n_steps_predict, hidden_sizes))
dir_names = ["{}nstepspredict_{}hdsz".format(*v) for v in values]
keys = [("algo", "n_steps_predict"), ("algo", "hidden_sizes")]
variant_levels.append(VariantLevel(keys, values, dir_names))

kl_losses = [1., 0.1]
values = list(zip(kl_losses))
dir_names = ["{}klcoef".format(*v) for v in values]
keys = [("algo", "kl_coeff")]
variant_levels.append(VariantLevel(keys, values, dir_names))



replay_base_dir = "/data/adam/ul4rl/replays/20200608/15M_VecEps_B78"
# games = [
#     "pong", "qbert", "seaquest", "space_invaders",
#     "alien", "breakout", "frostbite", "gravitar",
# ]
games = ["breakout", "gravitar", "qbert", "space_invaders"]
replay_filenames = [osp.join(replay_base_dir, game, "run_0/replaybuffer.pkl")
    for game in games]
values = list(zip(replay_filenames, games))
dir_names = games
keys = [("algo", "replay_filepath"), ("name",)]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

num_variants = len(variants)
variants_per = num_variants // num_computers

my_start = my_computer * variants_per
if my_computer == num_computers - 1:
    my_end = num_variants
else:
    my_end = (my_computer + 1) * variants_per
my_variants = variants[my_start:my_end]
my_log_dirs = log_dirs[my_start:my_end]

default_config_key = "basic_vae"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=my_variants,
    log_dirs=my_log_dirs,
    common_args=(default_config_key,),
)
