
import copy

configs = dict()


config = dict(
    agent=dict(
        store_latent=False,  # only if model stop_conv_grad=True
    ),
    algo=dict(
        discount=0.99,
        learning_rate=2.5e-4,
        value_loss_coeff=0.5,
        entropy_loss_coeff=0.01,  # LEVEL-SPECIFIC
        clip_grad_norm=100.,
        initial_optim_state_dict=None,
        gae_lambda=0.97,
        minibatches=2,
        epochs=1,
        ratio_clip=0.1,
        linear_lr_schedule=False,
        normalize_advantage=False,
        min_steps_rl=0,
        min_steps_ul=0,
        max_steps_ul=None,
        ul_learning_rate=0.001,
        ul_optim_kwargs=None,
        ul_replay_size=1e5,
        ul_update_schedule="constant_1",
        ul_lr_schedule=None,
        ul_lr_warmup=0,
        ul_delta_T=3,
        ul_batch_B=512,
        ul_batch_T=1,
        ul_random_shift_prob=1.,
        ul_random_shift_pad=4,
        ul_target_update_interval=1,
        ul_target_update_tau=0.01,
        ul_latent_size=256,
        ul_anchor_hidden_sizes=512,
        ul_clip_grad_norm=10.,
        ul_pri_alpha=0.,
        ul_pri_beta=1.,
        ul_pri_n_step_return=1,
    ),
    env=dict(
        level="lasertag_one_opponent_small",
        frame_history=1,
        fps=None,
    ),
    # Will use same args for eval env.
    model=dict(
        skip_connections=True,
        lstm_size=256,
        hidden_sizes=None,
        kiaming_init=True,
        stop_conv_grad=False,
        skip_lstm=True,
    ),
    optim=dict(),
    runner=dict(
        n_steps=25e6,
        log_interval_steps=1e5,
    ),
    sampler=dict(
        batch_T=128,
        batch_B=16,
        max_decorrelation_steps=3000,
    ),
)

configs["ppo_ul_16env"] = config
