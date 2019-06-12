
configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=1e-3,
        value_loss_coeff=1.,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
        gae_lambda=0.98,
        linear_lr_schedule=True,
    ),
    env=dict(
        game="pong",
        num_img_obs=1,
    ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=50e6,
        # log_interval_steps=1e3,
    ),
    sampler=dict(
        batch_T=64,
        batch_B=32,
        max_decorrelation_steps=1000,
    ),
)

configs["0"] = config
