
import copy

configs = dict()


config = dict(
    agent=dict(
        state_dict_filename=None,
        load_conv=True,
        load_all=False,  # Just for replay saving.
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
    ),
    env=dict(
        level="lasertag_one_opponent_small",
        frame_history=1,
        fps=None,
    ),
    # Will use same args for eval env.
    model=dict(
        # use_fourth_layer=True,
        skip_connections=True,
        lstm_size=256,
        hidden_sizes=None,
        kiaming_init=True,
        skip_lstm=True,
        stop_conv_grad=False,
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
    pretrain=dict(  # Just for logging purposes.
        name=None,
        algo=None,
        n_updates=None,
        log_interval_updates=None,
        learning_rate=None,
        target_update_tau=None,
        batch_B=None,
        batch_T=None,
        warmup_T=None,
        delta_T=None,
        hidden_sizes=None,
        latent_size=None,
        batch_size=None,
        validation_batch_size=None,
        activation_loss_coefficient=None,
        replay=None,
        model_dir=None,
        learning_rate_anneal=None,
        learning_rate_warmup=None,
        weight_decay=None,
        anchor_hidden_sizes=None,
        action_condition=False,
        transform_hidden_sizes=None,
        kiaming_init=True,
        data_aug=None,
        random_shift_prob=None,
        use_global_global=None,
        use_global_local=None,
        use_local_local=None,
        local_conv_layer=None,
    ),
)

configs["ppo_16env"] = config
