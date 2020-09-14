
import copy

configs = dict()

config = dict(
    agent=dict(
        action_squash=1.,
        pretrain_std=0.75,  # 0.75 gets pretty uniform actions
        load_conv=True,
        load_all=False,
        store_latent=False,
        state_dict_filename=None,
    ),
    conv=dict(
        channels=[32, 32, 32, 32],
        kernel_sizes=[3, 3, 3, 3],
        strides=[2, 2, 2, 1],
        paddings=None,
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
        min_steps_learn=int(1e4),
        replay_size=int(1e5),
        target_update_tau=0.01,  # tau=1 for hard update.
        target_update_interval=2,
        actor_update_interval=2,
        # OptimCls=torch.optim.Adam,
        initial_optim_state_dict=None,  # for all of them.
        action_prior="uniform",  # or "gaussian"
        reward_scale=1,
        target_entropy="auto",  # "auto", float, or None
        reparameterize=True,
        clip_grad_norm=1e6,
        n_step_return=1,
        # updates_per_sync=1,  # For async mode only.
        bootstrap_timelimit=True,
        # crop_size=84,  # Get from agent.
        q_lr=1e-3,
        pi_lr=1e-3,
        alpha_lr=1e-4,
        q_beta=0.9,
        pi_beta=0.9,
        alpha_beta=0.5,
        alpha_init=0.1,
        encoder_update_tau=0.05,
        augmentation="random_shift",  # [None, "random_shift", "subpixel_shift"]
        random_shift_pad=4,  # how much to pad on each direction (like DrQ style)
        random_shift_prob=1.,
        stop_conv_grad=False,
        max_pixel_shift=1.,
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
    pretrain=dict(  # Populate these for logging, to compare
        name=None,
        algo=None,
        n_updates=None,
        batch_size=None,
        batch_T=None,
        batch_B=None,
        delta_T=None,
        learning_rate=None,
        target_update_tau=None,
        target_update_interval=None,
        replay=None,
        model_dir=None,
        clip_grad_norm=None,
        activation_loss_coefficient=None,
        learning_rate_anneal=None,
        learning_rate_warmup=None,
        data_aug=None,
        random_shift_pad=None,
        random_shift_prob=None,
        latent_size=None,
        anchor_hidden_sizes=None,
        hidden_sizes=None,
        kl_coeff=None,
        weight_decay=None,
        kiaming_init=None,
        run_ID=0,
        log_interval_updates=None,
    )
)

configs["serial_radsac"] = config

config = copy.deepcopy(configs["serial_radsac"])
config["agent"]["load_conv"] = False
config["algo"]["min_steps_learn"] = 5e3
config["runner"]["n_steps"] = 50e3
config["algo"]["replay_size"] = 50e3
config["sampler"]["eval_max_steps"] = 30e3
config["sampler"]["eval_max_trajectories"] = 30
configs["replaysave"] = config
