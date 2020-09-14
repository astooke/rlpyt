
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

script = "rlpyt/ul/experiments/rl_with_ul/scripts/dmcontrol/train/dmc_sac_with_ul_serial.py"

affinity_code = quick_affinity_code(contexts_per_gpu=3)
runs_per_setting = 3
experiment_title = "sac_with_ul_final_1"


###############################################
# PRETRAIN CONFIG: MATCHES WHAT WAS ALREADY DONE

variant_levels_1 = list()
# variant_levels_2 = list()

min_steps_rl = [1e4]
min_steps_ul = [1e4]
ul_anneals = ["cosine"]
values = list(zip(min_steps_rl, min_steps_ul, ul_anneals))
dir_names = ["{}rlminstepsul{}_{}anneal".format(*v) for v in values]
keys = [("algo", "min_steps_rl"), ("algo", "min_steps_ul"),
    ("algo", "ul_lr_schedule")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))


stop_conv_grads = [False, True]
ul_updates = ["constant_0", "constant_1"]
values = list(zip(stop_conv_grads, ul_updates))
dir_names = ["{}stpcnvgrd_ul_{}".format(*v) for v in values]
keys = [("algo", "stop_rl_conv_grad"), ("algo", "ul_update_schedule")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))


doms = [
    "cheetah",
    "ball_in_cup",
    "cartpole", "cartpole",
    "walker",
    "hopper",
]
tasks = [
    "run",
    "catch",
    "swingup", "swingup_sparse",
    "walk",
    "hop",
]
fskips = [
    4,
    4,
    8, 8,
    2,
    4,
]
n_after_cheetah = len(doms) - 1
qlrs = [2e-4] + [1e-3] * n_after_cheetah
pilrs = [2e-4] + [1e-3] * n_after_cheetah
bss = [512] + [256] * n_after_cheetah  # [512]
ul_bss = [512] + [256] * n_after_cheetah
# rprs = [512] + [256] * 2  # [512]
# steps = [150e3, 150e3, 75e3, 3e5]
steps = [
    250e3,
    75e3,
    375e2, 125e3,
    250e3,
    250e3,
]
steps = [s + 1e4 for s in steps]  # 1e4 initialization min steps learn
values = list(zip(doms, tasks, fskips, qlrs, pilrs, bss, ul_bss, steps))
dir_names = [f"{dom}_{task}" for (dom, task) in zip(doms, tasks)]
keys = [
    ("env", "domain_name"),
    ("env", "task_name"),
    ("env", "frame_skip"),
    ("algo", "q_lr"),
    ("algo", "pi_lr"),
    ("algo", "batch_size"),
    # ("algo", "replay_ratio"),
    ("algo", "ul_batch_size"),
    ("runner", "n_steps"),
]
variant_levels_1.append(VariantLevel(keys, values, dir_names))
# variant_levels_2.append(VariantLevel(keys, values, dir_names))


variants_1, log_dirs_1 = make_variants(*variant_levels_1)
# variants_2, log_dirs_2 = make_variants(*variant_levels_2)

variants = variants_1  # + variants_2  # + variants_3
log_dirs = log_dirs_1  # + log_dirs_2  # + log_dirs_3


num_variants = len(variants)
variants_per = num_variants // num_computers

my_start = my_computer * variants_per
if my_computer == num_computers - 1:
    my_end = num_variants
else:
    my_end = (my_computer + 1) * variants_per
my_variants = variants[my_start:my_end]
my_log_dirs = log_dirs[my_start:my_end]

default_config_key = "sac_with_ul"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=my_variants,
    log_dirs=my_log_dirs,
    common_args=(default_config_key,),
)
