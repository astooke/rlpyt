
import copy

configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=3e-4,
        value_loss_coeff=0.5,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
    ),
    env=dict(
        game="pong",
        num_img_obs=1,
        ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=50e6,
        # log_interval_steps=1e5,
    ),
    sampler=dict(
        batch_T=20,
        batch_B=32,
        max_decorrelation_steps=1000,
    ),
)

configs["0"] = config


config = copy.deepcopy(config)
config["env"]["num_img_obs"] = 4
config["sampler"]["batch_T"] = 5
config["sampler"]["batch_B"] = 16
config["algo"]["learning_rate"] = 1e-4
configs["4frame"] = config


config = copy.deepcopy(config)
config["algo"]["learning_rate"] = 7e-4
config["sampler"]["batch_B"] = 32
config["algo"]["clip_grad_norm"] = 1
configs["like_ff"] = config
