
default_configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=7e-4,
        value_loss_coeff=0.5,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
    ),
    env=dict(),
    model=dict(),
    optim=dict(),
    runner=dict(n_steps=10e6),
    sampler=dict(
        batch_T=5,
        batch_B=64,
        max_path_length=27000,
        max_decorrelation_steps=1000,
    ),
)

default_configs["0"] = config
