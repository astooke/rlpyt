
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

script = "rlpyt/ul/experiments/rl_with_ul/scripts/atari/train/atari_ppo_with_ul_serial.py"

affinity_code = quick_affinity_code(contexts_per_gpu=4)
runs_per_setting = 4
experiment_title = "atari_ppo_with_ul_final_1"

variant_levels_1 = list()
# variant_levels_2 = list()
# variant_levels_3 = list()

ul_learning_rates = [1e-3]
ul_lr_anneals = ["cosine"]
rl_grad_norms = [1e4]
values = list(zip(ul_learning_rates, ul_lr_anneals, rl_grad_norms))
dir_names = ["final_atari_ul"]
keys = [("algo", "ul_learning_rate"), ("algo", "ul_lr_schedule"),
    ("algo", "clip_grad_norm")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))


stop_conv_grads = [False, False, False, True]
ul_schedules = ["constant_0", "front_10000_0", "quadratic_6", "quadratic_6"]
min_steps_ul = [0, 1e5, 5e4, 5e4]
min_steps_rl = [0, 1e5, 1e5, 1e5]
values = list(zip(stop_conv_grads, ul_schedules, min_steps_ul, min_steps_rl))
dir_names = ["RL", "RL_UL_init", "RL_UL", "UL"]
keys = [("model", "stop_conv_grad"), ("algo", "ul_update_schedule"),
    ("algo", "min_steps_ul"), ("algo", "min_steps_rl")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))


# games = ["pong", "qbert", "seaquest", "space_invaders",
#     "alien", "breakout", "frostbite", "gravitar"]
games = ["pong", "qbert", "seaquest", "space_invaders",
    "alien", "breakout", "ms_pacman", "gravitar"]
values = list(zip(games))
dir_names = games
keys = [("env", "game")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))
# variant_levels_2.append(VariantLevel(keys, values, dir_names))
# variant_levels_3.append(VariantLevel(keys, values, dir_names))


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

default_config_key = "ppo_ul_16env"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=my_variants,
    log_dirs=my_log_dirs,
    common_args=(default_config_key,),
)

