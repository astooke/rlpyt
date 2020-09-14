
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

script = "rlpyt/ul/experiments/rl_from_ul/scripts/dmlab/train/dmlab_ppo_from_ul_alt.py"

affinity_code = quick_affinity_code(contexts_per_gpu=1, alternating=True)
runs_per_setting = 2
experiment_title = "dmlab_ppo_from_cpc_2"
variant_levels_1 = list()
# variant_levels_2 = list()

learning_rates = [3e-4]  # trying lower learning rate and...
values = list(zip(learning_rates))
dir_names = ["{}lr".format(*v) for v in values]
keys = [("pretrain", "learning_rate")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))

n_updates = [50e3, 150e3]  # ..longer training
values = list(zip(n_updates))
dir_names = ["{}updates".format(*v) for v in values]
keys = [("pretrain", "n_updates")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))


batch_Bs = [8, 16, 32]
batch_Ts = [64, 32, 16]
warmup_Ts = [16, 16, 16]
values = list(zip(batch_Bs, batch_Ts, warmup_Ts))
dir_names = ["{}B_{}T_{}wmpT".format(*v) for v in values]
keys = [("pretrain", "batch_B"), ("pretrain", "batch_T"), ("pretrain", "warmup_T")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))


levels = [
    "lasertag_three_opponents_small",
    # "rooms_watermaze",
    "explore_goal_locations_small",
]
entropies = [
    0.0003,
    # 0.001,
    0.01,
]
values = list(zip(levels, entropies))
dir_names = levels
keys = [("env", "level"), ("algo", "entropy_loss_coeff")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))
# variant_levels_2.append(VariantLevel(keys, values, dir_names))


##################################################
# RL CONFIG (mostly)

##################################################
# RL CONFIG (mostly)

n_steps = [25e6]
pretrain_algos = ["CPC"]
replays = ["20200807/dmlab_replaysave_1"]
model_dirs = ["/data/adam/ul4rl/models/20200901/dmlab_cpc_pretrain_2/"]
values = list(zip(
    n_steps, 
    pretrain_algos, 
    replays, 
    model_dirs,
))
dir_names = ["RlFromUl"]  # TRAIN SCRIPT SPLITS OFF THIS
keys = [("runner", "n_steps"),
    ("pretrain", "algo"), 
    ("pretrain", "replay"),
    ("pretrain", "model_dir"), 
]
variant_levels_1.append(VariantLevel(keys, values, dir_names))


stop_grads = [True]
values = list(zip(stop_grads))
dir_names = ["{}stpcnvgrd".format(*v) for v in values]
keys = [("model", "stop_conv_grad")]
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
