
import copy

configs = dict()


config = dict(
    agent=dict(
        state_dict_filename=None,
        load_conv=True,
        store_latent=False,
    ),
    algo=dict(
        discount=0.99,
        learning_rate=2.5e-4,
        value_loss_coeff=1.,
        entropy_loss_coeff=0.01,
        clip_grad_norm=10.,
        initial_optim_state_dict=None,
        gae_lambda=0.95,
        minibatches=4,
        epochs=4,
        ratio_clip=0.1,
        linear_lr_schedule=True,
        normalize_advantage=False,
    ),
    env=dict(
        game="pong",
        episodic_lives=False,  # new standard
        repeat_action_probability=0.25,  # sticky actions
        horizon=int(27e3),
    ),
    # Will use same args for eval env.
    model=dict(
        channels=None,
        kernel_sizes=None,
        strides=None,
        hidden_sizes=512,
        paddings=None,  # No padding for standard 84x84 images.
        stop_conv_grad=False,
        kiaming_init=True,
        normalize_conv_out=False,
    ),
    optim=dict(),
    runner=dict(
        n_steps=25e6,
        log_interval_steps=1e5,
    ),
    sampler=dict(
        batch_T=128,
        batch_B=16,
        max_decorrelation_steps=1000,
    ),
    pretrain=dict(  # Just for logging purposes.  Pile them all in here.
        name=None,
        algo=None,
        n_updates=None,
        log_interval_updates=None,
        learning_rate=None,
        target_update_tau=None,
        positives=None,
        replay_sequence_length=None,
        hidden_sizes=None,
        latent_size=None,
        batch_size=None,
        validation_batch_size=None,
        activation_loss_coefficient=None,
        replay=None,
        model_dir=None,
        learning_rate_anneal=None,
        learning_rate_warmup=None,
        weight_decay=0,
        action_condition=None,
        kiaming_init=True,
        data_aug=None,
        use_global_global=None,
        use_global_local=None,
        use_local_local=None,
        local_conv_layer=None,
        delta_T=None,
        batch_B=None,
        batch_T=None,
        anchor_hidden_sizes=None,
        random_shift_prob=None,
        random_shift_pad=None,
        entropy_loss_coeff=None,
        onehot_action=None,
        kl_coeff=None,
    ),
)

configs["ppo_16env"] = config
