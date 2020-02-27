
import copy

configs = dict()

config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=3e-5,
        clip_grad_norm=1e6,
        entropy_loss_coeff=0.0,
        value_loss_coeff=0.5,
        normalize_advantage=True,
    ),
    env=dict(id="Hopper-v3"),
    model=dict(normalize_observation=False),
    optim=dict(),
    runner=dict(
        n_steps=1e6,
        log_interval_steps=2e4,
    ),
    sampler=dict(
        batch_T=100,
        batch_B=8,
        max_decorrelation_steps=1000,
    ),
)

configs["a2c_1M"] = config


