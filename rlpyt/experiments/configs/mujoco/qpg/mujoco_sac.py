
import copy

configs = dict()

config = dict(
    agent=dict(
        q_model_kwargs=None,
        v_model_kwargs=None,
        pi_model_kwargs=None,
    ),
    algo=dict(
        discount=0.99,
        batch_size=256,
        training_ratio=256,
        target_update_tau=0.005,
        target_update_interval=1,
        learning_rate=3e-4,
        reparameterize=True,
        policy_output_regularization=0.001,
        reward_scale=5
    ),
    env=dict(id="Hopper-v3"),
    model=dict(),
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

configs["sac_1M_serial"] = config