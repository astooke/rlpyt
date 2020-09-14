
import copy

configs = dict()

config = dict(
    agent=dict(
        action_squash=1.,
        pretrain_std=0.75,  # 0.75 gets pretty uniform actions
    ),
    conv=dict(
        channels=[32, 32, 32, 32],
        kernel_sizes=[3, 3, 3, 3],
        strides=[2, 2, 2, 1],
        paddings=None,
        final_nonlinearity=True,
    ),
    fc1=dict(
        latent_size=50,
        layer_norm=True,
    ),
    pi_model=dict(
        hidden_sizes=[1024, 1024],
        min_log_std=-10,
        max_log_std=2,
    ),
    q_model=dict(hidden_sizes=[1024, 1024]),
    algo=dict(
        discount=0.99,
        batch_size=512,
        # replay_ratio=512,  # data_consumption / data_generation
        # min_steps_learn=int(1e4),
        replay_size=int(1e5),
        target_update_tau=0.01,  # tau=1 for hard update.
        target_update_interval=2,
        actor_update_interval=2,
        initial_optim_state_dict=None,  # for all of them.
        action_prior="uniform",  # or "gaussian"
        reward_scale=1,
        target_entropy="auto",  # "auto", float, or None
        reparameterize=True,
        clip_grad_norm=1e6,
        n_step_return=1,
        # updates_per_sync=1,  # For async mode only.
        bootstrap_timelimit=True,
        q_lr=1e-3,
        pi_lr=1e-3,
        alpha_lr=1e-4,
        q_beta=0.9,
        pi_beta=0.9,
        alpha_beta=0.5,
        alpha_init=0.1,
        encoder_update_tau=0.05,
        random_shift_prob=1.,
        random_shift_pad=4,  # how much to pad on each direction (like DrQ style)
        stop_rl_conv_grad=False,
        min_steps_rl=int(1e4),
        min_steps_ul=int(1e4),
        max_steps_ul=None,
        ul_learning_rate=1e-3,
        ul_optim_kwargs=None,
        # ul_replay_size=1e5,
        ul_update_schedule="constant_1",
        ul_lr_schedule=None,
        ul_lr_warmup=0,
        ul_batch_size=512,
        ul_random_shift_prob=1.,
        ul_random_shift_pad=4,
        ul_target_update_interval=1,
        ul_target_update_tau=0.01,
        ul_latent_size=128,
        ul_anchor_hidden_sizes=512,
        ul_clip_grad_norm=10.,
        ul_pri_alpha=0.,
        ul_pri_beta=1.,
        ul_pri_n_step_return=1,
        ul_use_rl_samples=False,
    ),
    env=dict(
        domain_name="cheetah",
        task_name="run",
        from_pixels=True,
        frame_stack=3,
        frame_skip=4,
        height=84,
        width=84,
    ),
    optim=dict(),
    runner=dict(
        n_steps=1e5,
        log_interval_steps=1e3,
    ),
    sampler=dict(
        batch_T=1,
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=5,
        eval_max_steps=int(10000),
        eval_max_trajectories=10,
    ),
)

configs["sac_with_ul"] = config
