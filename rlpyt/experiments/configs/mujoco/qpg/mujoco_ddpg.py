
import copy

configs = dict()

config = dict(
    agent=dict(
        model_kwargs=None,
        q_model_kwargs=None,
    ),
    algo=dict(
        discount=0.99,
        batch_size=100,
        replay_ratio=100,
        target_update_tau=0.01,
        target_update_interval=1,
        policy_update_interval=1,
        learning_rate=1e-3,
        q_learning_rate=1e-3,
    ),
    env=dict(id="Hopper-v3"),
    # eval_env=dict(id="Hopper-v3"),  # Same kwargs as env, in train script.
    optim=dict(),
    runner=dict(
        n_steps=1e6,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=1,
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=5,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
    ),
)

configs["ddpg_from_td3_1M_serial"] = config

config = copy.deepcopy(config)
config["sampler"]["batch_T"] = 5
config["algo"]["updates_per_sync"] = 1
configs["async_serial"] = config
