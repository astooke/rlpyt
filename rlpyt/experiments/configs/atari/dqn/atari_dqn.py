
import copy

configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        batch_size=128,
        learning_rate=1e-4,  # Trying lower for adam.
        clip_grad_norm=10.,
        min_steps_learn=int(5e4),
        double_dqn=False,
        prioritized_replay=False,
        n_step_return=1,
        replay_size=int(1e6),
    ),
    env=dict(
        game="pong",
        episodic_lives=True,
    ),
    eval_env=dict(
        game="pong",  # NOTE: update in train script!
        episodic_lives=False,
        horizon=int(27e3),
    ),
    model=dict(dueling=False),
    optim=dict(),
    runner=dict(
        n_steps=50e6,
        log_interval_steps=1e6,
    ),
    sampler=dict(
        batch_T=2,
        batch_B=16,
        max_decorrelation_steps=1000,
        eval_n_envs=4,
        eval_max_steps=int(125e3),
        eval_max_trajectories=100,
    ),
)

configs["dqn"] = config

config = copy.deepcopy(configs["dqn"])
config["algo"]["double_dqn"] = True
configs["double"] = config

config = copy.deepcopy(configs["dqn"])
config["algo"]["prioritized_replay"] = True
configs["prioritized"] = config

config = copy.deepcopy(configs["dqn"])
config["model"]["dueling"] = True
configs["dueling"] = config


config = copy.deepcopy(configs["dqn"])
config["algo"]["double_dqn"] = True
config["algo"]["prioritized_replay"] = True
config["model"]["dueling"] = True
configs["double_pri_duel"] = config


config = copy.deepcopy(configs["dqn"])
config["algo"]["learning_rate"] = 2.5e-4
configs["catdqn"] = config

config = copy.deepcopy(configs["dqn"])
config["algo"]["n_step_return"] = 3
config["algo"]["learning_rate"] = 6.25e-5
config["algo"]["double_dqn"] = True
config["algo"]["prioritized_replay"] = True
config["model"]["dueling"] = True
configs["ernbw"] = config


config = copy.deepcopy(configs["dqn"])
config["optim"] = dict(alpha=0.95, eps=1e-6)
config["algo"]["learning_rate"] = 2.5e-4
config["runner"]["n_steps"] = 15e6
configs["rmsprop"] = config


config = copy.deepcopy(configs["dqn"])
config["runner"]["n_steps"] = 15e6
config["runner"]["log_interval_steps"] = 1e5
config["sampler"]["eval_max_steps"] = 70e3
config["sampler"]["eval_max_trajectories"] = 50
configs["fast_log"] = config

config = copy.deepcopy(configs["dqn"])
config["runner"]["n_steps"] = 15e6
configs["short_run"] = config


config = copy.deepcopy(configs["dqn"])
config["sampler"]["eval_n_envs"] = 0
config["runner"]["log_interval_steps"] = int(1e5)
configs["no_eval"] = config


config = copy.deepcopy(configs["dqn"])
config["sampler"]["batch_T"] = 8
config["sampler"]["batch_B"] = 2
configs["serial"] = config

config = copy.deepcopy(configs["dqn"])
config["sampler"]["batch_T"] = 4
config["sampler"]["batch_B"] = 4
configs["cpu"] = config

config = copy.deepcopy(configs["dqn"])
config["sampler"]["batch_T"] = 8
config["sampler"]["batch_B"] = 2
config["sampler"]["max_decorrelation_steps"] = 20
config["sampler"]["eval_max_steps"] = 8e3
config["sampler"]["eval_max_trajectories"] = 4
config["runner"]["n_steps"] = 3e4
config["runner"]["log_interval_steps"] = 1e4
config["algo"]["min_steps_learn"] = 1e4
config["algo"]["replay_size"] = 1e4
config["algo"]["eps_steps"] = 2e4
configs["debug"] = config

config = copy.deepcopy(configs["debug"])
config["sampler"]["batch_T"] = 4
config["sampler"]["batch_B"] = 4
config["agent"]["eps_final_min"] = 0.001
configs["debug_vec_eps"] = config

config = copy.deepcopy(configs["dqn"])
config["sampler"]["batch_T"] = 4
config["sampler"]["batch_B"] = 256
config["algo"]["prioritized_replay"] = False
config["eval_n_envs"] = 10
configs["async_big"] = config
