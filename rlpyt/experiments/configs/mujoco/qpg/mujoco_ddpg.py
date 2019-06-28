
import copy

configs = dict()

config = dict(
    agent=dict(
        mu_model_kwargs=None,
        q_model_kwargs=None,
    ),
    algo=dict(
        discount=0.99,
        batch_size=100,
        training_ratio=100,
        target_update_tau=0.01,
        target_update_interval=1,
        policy_update_interval=1,
        mu_learning_rate=1e-3,
        q_learning_rate=1e-3,
    ),
    env=dict(id="Hopper-v3"),
    optim=dict(),
    runner=dict(
        n_steps=1e6,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=1,
        batch_B=1,
        max_decorrelation_steps=1000,
    ),
)

configs["ddpg_from_td3_1M_serial"] = config