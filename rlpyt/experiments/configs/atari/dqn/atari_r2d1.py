
# import copy

configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.997,
        batch_T=80,
        batch_B=32,  # In the paper, 64.
        warmup_T=40,
        store_rnn_state_interval=40,
        training_ratio=4,  # In the paper, more like 0.8.
        learning_rate=1e-4,
        clip_grad_norm=10.,
        min_steps_learn=int(1e5),
        double_dqn=True,
        prioritized_replay=True,
        n_step_return=5,
    ),
    env=dict(
        game="pong",
        episodic_lives=True,  # The paper does mostly without, but still better.
        clip_reward=False,
        horizon=int(40e3),
    ),
    eval_env=dict(
        game="pong",  # NOTE: update in train script!
        episodic_lives=False,
        horizon=int(40e3),
        clip_reward=False,
    ),
    model=dict(dueling=True),
    optim=dict(),
    runner=dict(
        n_steps=100e6,
        log_interval_steps=1e6,
    ),
    sampler=dict(
        batch_T=30,  # Match the algo / training_ratio.
        batch_B=32,
        max_decorrelation_steps=1000,
        eval_n_envs=4,
        eval_max_steps=int(161e3),  # DEBUG
        eval_max_trajectories=100,
        eval_min_envs_reset=2,
    ),
)

configs["r2d1"] = config

