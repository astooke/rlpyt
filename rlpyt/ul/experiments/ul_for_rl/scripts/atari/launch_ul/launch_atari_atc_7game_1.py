
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

script = "rlpyt/ul/experiments/ul_for_rl/scripts/atari/train_ul/atari_atc.py"

affinity_code = quick_affinity_code(contexts_per_gpu=2)
runs_per_setting = 1
experiment_title = "atari_atc_ul_7game_1"
variant_levels = list()

n_updates = [50e3]
learning_rates = [1e-3]
values = list(zip(n_updates, learning_rates))
dir_names = ["{}updates_{}lr".format(*v) for v in values]
keys = [("runner", "n_updates"), ("algo", "learning_rate")]
variant_levels.append(VariantLevel(keys, values, dir_names))

replay_base_dir = "/data/adam/ul4rl/replays/20200608/15M_VecEps_B78"
games = [
    "pong", "qbert", "seaquest", "space_invaders",
    "alien", "breakout", "frostbite", "gravitar",
]
# Make a round robin where one game is left out.
replay_filenames = list()
names = list()
for g in range(len(games)):
    sub_games = games.copy()
    holdout = sub_games.pop(g)  # remove one
    filenames = [osp.join(replay_base_dir, game, "run_0/replaybuffer.pkl")
        for game in sub_games]
    replay_filenames.append(filenames)
    names.append(holdout + "_holdout")
values = list(zip(replay_filenames, names))
dir_names = names
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

default_config_key = "atari_atc"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=my_variants,
    log_dirs=my_log_dirs,
    common_args=(default_config_key,),
)
