
import sys
import copy
import os.path as osp

from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

args = sys.argv[1:]
assert len(args) == 2 or len(args) == 0
if len(args) == 0:
    my_computer, num_computers = 0, 1
else:
    my_computer = int(args[0])
    num_computers = int(args[1])

print(f"MY_COMPUTER: {my_computer},  NUM_COMPUTERS: {num_computers}")

script = "rlpyt/ul/experiments/ul_for_rl/scripts/dmlab/train_ul/dmlab_cpc.py"

affinity_code = quick_affinity_code(contexts_per_gpu=1)
runs_per_setting = 1
experiment_title = "dmlab_cpc_pretrain_1"
variant_levels_1 = list()
# variant_levels_2 = list()

n_updates = [20e3, 40e3]
values = list(zip(n_updates))
dir_names = ["{}updates".format(*v) for v in values]
keys = [("runner", "n_updates")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))


batch_Bs = [8, 16, 32]
batch_Ts = [64, 32, 16]
warmup_Ts = [16, 16, 16]
values = list(zip(batch_Bs, batch_Ts, warmup_Ts))
dir_names = ["{}B_{}T_{}wmpT".format(*v) for v in values]
keys = [("algo", "batch_B"), ("algo", "batch_T"), ("algo", "warmup_T")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))


replay_base_dir = "/data/adam/ul4rl/replays/20200807/dmlab_replaysave_1"
levels = [
    "lasertag_three_opponents_small",
    "explore_goal_locations_small",
    # "rooms_watermaze",
]
replay_filenames = [osp.join(replay_base_dir, level, "run_0/replaybuffer.pkl")
    for level in levels]
values = list(zip(replay_filenames, levels))
dir_names = levels
keys = [("algo", "replay_filepath"), ("name",)]
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

default_config_key = "dmlab_cpc"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=my_variants,
    log_dirs=my_log_dirs,
    common_args=(default_config_key,),
)
