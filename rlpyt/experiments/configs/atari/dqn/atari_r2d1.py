
import copy

configs = dict()


config = dict(
    agent=dict(),
    model=dict(dueling=True),
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
    optim=dict(),
    env=dict(
        game="pong",
        episodic_lives=True,  # The paper does mostly without, but still better.
        clip_reward=False,
        horizon=int(27e3),
        num_img_obs=4,
    ),
    eval_env=dict(
        game="pong",  # NOTE: update in train script!
        episodic_lives=False,
        horizon=int(27e3),
        clip_reward=False,
        num_img_obs=4,
    ),
    runner=dict(
        n_steps=100e6,
        log_interval_steps=1e6,
    ),
    sampler=dict(
        batch_T=30,  # Match the algo / training_ratio.
        batch_B=32,
        max_decorrelation_steps=1000,
        eval_n_envs=4,
        eval_max_steps=int(161e3),
        eval_max_trajectories=100,
        # eval_min_envs_reset=2,
    ),
)

configs["r2d1"] = config


config = copy.deepcopy(configs["r2d1"])
config["algo"]["replay_size"] = int(4e6)
config["algo"]["batch_B"] = 64  # Not sure will fit.
config["algo"]["training_ratio"] = 1
config["algo"]["eps_final"] = 0.1
config["algo"]["eps_final_min"] = 0.0005
config["runner"]["n_steps"] = 20e9
config["runner"]["log_interval_steps"] = 10e6
config["sampler"]["batch_T"] = 40  # = warmup_T = store_rnn_interval; new traj at boundary.
config["sampler"]["batch_B"] = 192  # to make one update per sample batch.
config["sampler"]["eval_n_envs"] = 6  # 6 cpus, 6 * 32 = 192, for pabti.
config["sampler"]["eval_max_steps"] = int(28e3 * 6)
config["env"]["episodic_lives"] = False
configs["r2d1_long"] = config

