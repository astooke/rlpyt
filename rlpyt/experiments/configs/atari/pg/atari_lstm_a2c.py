
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
    env=dict(
        game="pong",
        num_img_obs=1,
        ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=10e6,
        # log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=20,
        batch_B=16,
        max_path_length=27000,
        max_decorrelation_steps=1000,
    ),
)

configs["0"] = config
