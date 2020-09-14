
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

script = "rlpyt/ul/experiments/rl_from_ul/scripts/dmcontrol/train/dmc_sac_from_ul_serial.py"

affinity_code = quick_affinity_code(contexts_per_gpu=3)
runs_per_setting = 4
experiment_title = "dmc_sac_from_vaae_1"


###############################################
# PRETRAIN CONFIG: MATCHES WHAT WAS ALREADY DONE

variant_levels = list()
# variant_levels_2 = list()



n_updates = [1e4, 5e4]
values = list(zip(n_updates))
dir_names = ["{}updates".format(*v) for v in values]
keys = [("pretrain", "n_updates")]
variant_levels.append(VariantLevel(keys, values, dir_names))

delta_Ts = [0, 1]
hidden_sizes = [None, 512]
values = list(zip(delta_Ts, hidden_sizes))
dir_names = ["{}deltaT_{}hdsz".format(*v) for v in values]
keys = [("pretrain", "delta_T"), ("pretrain", "hidden_sizes")]
variant_levels.append(VariantLevel(keys, values, dir_names))

kl_losses = [1., 0.1]
values = list(zip(kl_losses))
dir_names = ["{}klcoef".format(*v) for v in values]
keys = [("pretrain", "kl_coeff")]
variant_levels.append(VariantLevel(keys, values, dir_names))




##################################################
# Some RL CONFIG (mostly)

doms = ["cheetah", "ball_in_cup", "cartpole", "walker"]
tasks = ["run", "catch", "swingup", "walk"]
fskips = [4, 4, 8, 2]
qlrs = [2e-4] + [1e-3] * 3
pilrs = [2e-4] + [1e-3] * 3
bss = [512] + [256] * 3  # [512]
# rprs = [512] * 4  # [512]
# steps = [150e3, 150e3, 75e3, 3e5]
steps = [150e3, 75e3, 375e2, 200e3]
steps = [s + 1e4 for s in steps]  # 1e4 initialization min steps learn
values = list(zip(doms, tasks, fskips, qlrs, pilrs, bss, steps))
dir_names = doms
keys = [
    ("env", "domain_name"),
    ("env", "task_name"),
    ("env", "frame_skip"),
    ("algo", "q_lr"),
    ("algo", "pi_lr"),
    ("algo", "batch_size"),
    # ("algo", "replay_ratio"),
    ("runner", "n_steps"),
]
variant_levels.append(VariantLevel(keys, values, dir_names))
# variant_levels_2.append(VariantLevel(keys, values, dir_names))


########################################################
# UL Config which was the same for all in the run.

pretrain_algos = ["VAE"]
replays = ["20200715/rad_sac_replaysave84"]
model_dirs = ["/data/adam/ul4rl/models/20200826/dmc_vae_pretrain_1/"]
values = list(zip(
    pretrain_algos, 
    replays, 
    model_dirs,
))
dir_names = ["RlFromUl"]  # TRAIN SCRIPT SPLITS OFF THIS
keys = [
    ("pretrain", "algo"), 
    ("pretrain", "replay"),
    ("pretrain", "model_dir"), 
]
variant_levels.append(VariantLevel(keys, values, dir_names))


################################################
# Rest of the RL config

stop_conv_grads = [True]
values = list(zip(stop_conv_grads))
dir_names = ["{}stopconvgrad".format(*v) for v in values]
keys = [("algo", "stop_conv_grad")]
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

default_config_key = "serial_radsac"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=my_variants,
    log_dirs=my_log_dirs,
    common_args=(default_config_key, experiment_title),
)
