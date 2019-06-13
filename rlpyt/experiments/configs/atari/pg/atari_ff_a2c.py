
configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=1e-3,
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
        batch_B=64,
        max_decorrelation_steps=1000,
    ),
)

configs["0"] = config
