
import copy

configs = dict()

config = dict(
    agent=dict(
        model_kwargs=None,
        q_model_kwargs=None,
        v_model_kwargs=None,
    ),
    algo=dict(
        discount=0.99,
        batch_size=256,
        replay_ratio=256,
        target_update_tau=0.005,
        target_update_interval=1,
        learning_rate=3e-4,
        reparameterize=True,
        policy_output_regularization=0.001,
        reward_scale=5,
    ),
    env=dict(id="Hopper-v3"),
    # eval_env=dict(id="Hopper-v3"),  # Train script uses "env".
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=1e6,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=1,
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=6,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
    ),
)

configs["sac_1M_serial"] = config

config = copy.deepcopy(configs["sac_1M_serial"])
config["algo"]["bootstrap_timelimit"] = True
configs["sac_serial_bstl"] = config

config = copy.deepcopy(config)
config["sampler"]["batch_T"] = 5
config["sampler"]["batch_B"] = 3
config["algo"]["updates_per_sync"] = 1
configs["async_gpu"] = config
