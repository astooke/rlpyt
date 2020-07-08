
from rlpyt.utils.launching.variant import make_variants, VariantLevel
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.affinity import quick_affinity_code


script = "rlpyt/projects/safe/experiments/scripts/train/train_cppo.py"

default_config_key = "LSTM"
affinity_code = quick_affinity_code()
runs_per_setting = 4
experiment_title = "PointGoal_Ki_Kp"

variant_levels = list()

env_ids = [
    # "Safexp-PointGoal0-v0",
    "Safexp-PointGoal1-v0",
    # "Safexp-PointGoal2-v0",
    # "Safexp-PointButton0-v0",
    # "Safexp-PointButton1-v0",
    # "Safexp-PointButton2-v0",
    # "Safexp-PointPush0-v0",
    # "Safexp-PointPush1-v0",
    # "Safexp-PointPush2-v0",
    # "Safexp-CarGoal0-v0",
    # "Safexp-CarGoal1-v0",
    # "Safexp-CarGoal2-v0",
    # "Safexp-CarButton0-v0",
    # "Safexp-CarButton1-v0",
    # "Safexp-CarButton2-v0",
    # "Safexp-CarPush0-v0",
    # "Safexp-CarPush1-v0",
    # "Safexp-CarPush2-v0",
    # "Safexp-DoggoGoal0-v0",
    # "Safexp-DoggoGoal1-v0",
    # "Safexp-DoggoGoal2-v0",
    # "Safexp-DoggoButton0-v0",
    # "Safexp-DoggoButton1-v0",
    # "Safexp-DoggoButton2-v0",
    # "Safexp-DoggoPush0-v0",
    # "Safexp-DoggoPush1-v0",
    # "Safexp-DoggoPush2-v0",
]

values = list(zip(env_ids))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "id")]
variant_levels.append(VariantLevel(keys, values, dir_names))

n_steps = [25e6]
cost_limits = [25]
values = list(zip(n_steps, cost_limits))
dir_names = ["{}steps_{}clim".format(*v) for v in values]
keys = [("runner", "n_steps"), ("algo", "cost_limit")]
variant_levels.append(VariantLevel(keys, values, dir_names))


pid_Kis = [1e-4, 1e-3, 1e-2, 1e-1, 1]
values = list(zip(pid_Kis))
dir_names = ["{}Ki".format(*v) for v in values]
keys = [("algo", "pid_Ki")]
variant_levels.append(VariantLevel(keys, values, dir_names))


pid_Kps = [0, 1]
values = list(zip(pid_Kps))
dir_names = ["{}Kp".format(*v) for v in values]
keys = [("algo", "pid_Kp")]
variant_levels.append(VariantLevel(keys, values, dir_names))

pid_delta_p_ema_alphas = [0.95]
values = list(zip(pid_delta_p_ema_alphas))
dir_names = ["{}dltpema".format(*v) for v in values]
keys = [("algo", "pid_delta_p_ema_alpha")]
variant_levels.append(VariantLevel(keys, values, dir_names))


variants, log_dirs = make_variants(*variant_levels)

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
