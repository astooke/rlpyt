
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

affinity_code = quick_affinity_code(contexts_per_gpu=3)
runs_per_setting = 2
experiment_title = "atari_ppo_with_ul_priority_1"

variant_levels_1 = list()
# variant_levels_2 = list()
# variant_levels_3 = list()

stop_conv_grads = [True]
rl_grad_norms = [1e4]
values = list(zip(stop_conv_grads, rl_grad_norms))
dir_names = ["{}stpcnvgrd_{}rlgrdnrm".format(*v) for v in values]
keys = [("model", "stop_conv_grad"), ("algo", "clip_grad_norm")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

min_steps_rl = [1e5]
values = list(zip(min_steps_rl))
dir_names = ["{}rlminsteps".format(*v) for v in values]
keys = [("algo", "min_steps_rl")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

ul_update_schedules = ["quadratic_15", "constant_4"]
max_steps_ul = [10e6, None]
min_steps_ul = [5e4, 5e4]
values = list(zip(ul_update_schedules, max_steps_ul, min_steps_ul))
dir_names = ["{}_{}maxstepulmin{}".format(*v) for v in values]
keys = [("algo", "ul_update_schedule"), ("algo", "max_steps_ul"),
    ("algo", "min_steps_ul")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

ul_lr_anneals = [None, "cosine"]
values = list(zip(ul_lr_anneals))
dir_names = ["{}anneal".format(*v) for v in values]
keys = [("algo", "ul_lr_schedule")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

ul_rs_probs = [0.1]
values = list(zip(ul_rs_probs))
dir_names = ["{}rsprob".format(*v) for v in values]
keys = [("algo", "ul_random_shift_prob")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

ul_pri_alphas = [1, 10]
values = list(zip(ul_pri_alphas))
dir_names = ["{}prialpha".format(*v) for v in values]
keys = [("algo", "ul_pri_alpha")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

ul_pri_n_steps = [1, 5, 20, 100]
values = list(zip(ul_pri_n_steps))
dir_names = ["{}prinstep".format(*v) for v in values]
keys = [("algo", "ul_pri_n_step_return")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))



# games = ["pong", "qbert", "seaquest", "space_invaders",
#     "alien", "breakout", "frostbite", "gravitar"]
# games = ["breakout", "gravitar", "qbert", "space_invaders"]
games = ["breakout", "space_invaders", "alien"]
values = list(zip(games))
dir_names = games
keys = [("env", "game")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))
# variant_levels_2.append(VariantLevel(keys, values, dir_names))
# variant_levels_3.append(VariantLevel(keys, values, dir_names))



##################################################
# RL CONFIG (mostly)

##################################################
# RL CONFIG (mostly)


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

