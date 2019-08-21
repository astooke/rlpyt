
import copy

configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=7e-4,
        value_loss_coeff=0.5,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
    ),
    env=dict(game="pong"),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=50e6,
        # log_interval_steps=1e3,
    ),
    sampler=dict(
        batch_T=5,
        batch_B=32,
        max_decorrelation_steps=1000,
    ),
)

configs["0"] = config

config = copy.deepcopy(config)

config["algo"]["learning_rate"] = 4e-4

configs["low_lr"] = config
