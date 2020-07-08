
import copy

configs = dict()

config = dict(
    env=dict(
        id="Safexp-PointGoal0-v0",
        obs_prev_cost=True,
        obs_version="default",
    ),
    sampler=dict(
        batch_T=128,
        batch_B=104,  # Might bust memory limits.
        max_decorrelation_steps=1000,
    ),
    algo=dict(
        discount=0.99,
        learning_rate=1e-4,
        value_loss_coeff=1.,
        entropy_loss_coeff=0,
        clip_grad_norm=1e4,
        gae_lambda=0.97,
        minibatches=1,
        epochs=8,
        ratio_clip=0.1,
        linear_lr_schedule=False,
        normalize_advantage=False,
        cost_discount=None,
        cost_gae_lambda=None,
        cost_value_loss_coeff=0.5,
        ep_cost_ema_alpha=0,  # 0 for hard update.
        objective_penalized=True,
        learn_c_value=True,
        penalty_init=0.,
        cost_limit=25,
        cost_scale=10,  # yes 10.
        normalize_cost_advantage=False,
        pid_Kp=0,
        pid_Ki=1,
        pid_Kd=0,
        pid_d_delay=1,
        pid_delta_p_ema_alpha=0.95,  # 0 for hard update
        pid_delta_d_ema_alpha=0.95,
        sum_norm=True,
        diff_norm=False,
        penalty_max=100,  # only if sum_norm=diff_norm=False
        step_cost_limit_steps=None,
        step_cost_limit_value=None,
        use_beta_kl=False,
        use_beta_grad=False,
        record_beta_kl=False,
        record_beta_grad=False,
        beta_max=10,
        beta_ema_alpha=0.9,
        beta_kl_epochs=1,
        reward_scale=1,
        lagrange_quadratic_penalty=False,
        quadratic_penalty_coeff=1,
    ),
    agent=dict(),
    model=dict(
        hidden_sizes=[512, 512],
        lstm_size=512,
        lstm_skip=True,
        constraint=True,  # must match algo.learn_c_value
        normalize_observation=True,
        var_clip=1e-6,
    ),
    runner=dict(
        n_steps=25e6,
        log_interval_steps=1e5,
    ),
)

configs["LSTM"] = config
