
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

script = "rlpyt/ul/experiments/ul_for_rl/scripts/dmcontrol/train_ul/dmc_atc.py"

affinity_code = quick_affinity_code(contexts_per_gpu=2)
runs_per_setting = 1
experiment_title = "dmc_ac_pretrain_1"
variant_levels_1 = list()
# variant_levels_2 = list()


delta_Ts = [0]  # No temporal contrast, just self augmented contrast
anchor_hidden_sizes = [None]  # No MLP on the anchor side
values = list(zip(delta_Ts, anchor_hidden_sizes))
dir_names = ["{}deltaT_{}anchhdsz".format(*v) for v in values]
keys = [("algo", "delta_T"), ("algo", "anchor_hidden_sizes")]
variant_levels_1.append(VariantLevel(keys, values, dir_names))


replay_base_dir = "/data/adam/ul4rl/replays/20200715/rad_sac_replaysave84"
domains = ["ball_in_cup", "cartpole", "cheetah", "walker"]
replay_filenames = [osp.join(replay_base_dir, game, "run_0/replaybuffer.pkl")
    for game in domains]
values = list(zip(replay_filenames, domains))
dir_names = domains
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

default_config_key = "dmc_atc"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=my_variants,
    log_dirs=my_log_dirs,
    common_args=(default_config_key,),
)
