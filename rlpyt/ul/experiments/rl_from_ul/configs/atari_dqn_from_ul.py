
import copy

configs = dict()


config = dict(
    agent=dict(
        state_dict_filename=None,
        load_conv=True,
        load_fc1=True,
    ),
    algo=dict(
        discount=0.99,
        batch_size=32,  # increase this first!
        learning_rate=6.25e-5,  # Adam Optimizer
        target_update_interval=2000,
        clip_grad_norm=10.,
        min_steps_learn=int(5e4),
        double_dqn=True,
        prioritized_replay=False,
        n_step_return=1,
        replay_size=int(1e6),
    ),
    env=dict(
        game="pong",
        episodic_lives=False,  # new standard
        repeat_action_probability=0.25,  # sticky actions
        horizon=int(27e3),
    ),
    # Will use same args for eval env.
    model=dict(
        # dueling=False,
        channels=None,
        kernel_sizes=None,
        strides=None,
        # paddings=None,
        fc1=512,
        hidden_sizes=None,
        paddings=[0, 0, 0],  # No padding for standard 84x84 images.
        use_maxpool=False,
        stop_grad=None,
        kiaming_init=True,
        final_conv_nonlinearity="relu",
    ),
    optim=dict(eps=0.01 / 32),  # REMEMBER TO SCALE WITH BATCH SIZE
    runner=dict(
        n_steps=50e6,
        log_interval_steps=1e6,
    ),
    sampler=dict(
        batch_T=4,
        batch_B=1,
        max_decorrelation_steps=1000,
        eval_n_envs=1,
        eval_max_steps=int(250e3),
        eval_max_trajectories=100,
    ),
    pretrain=dict(  # Just for logging purposes.
        name=None,
        algo=None,
        fc1_size=None,
        cpc_hidden_sizes=None,
        n_updates=None,
        learning_rate=None,
        target_update_tau=None,
        positives=None,
        replay_sequence_length=None,
        latent_size=None,
        batch_size=None,
        validation_batch_size=None,
        activation_loss_coefficient=None,
        replay=None,
        model_dir=None,
        learning_rate_anneal=None,
        learning_rate_warmup=None,
        weight_decay=0,
        anchor_mlp=False,
        action_condition=False,
        transform_hidden_sizes=None,
        kiaming_init=True,
        data_aug=None,
        random_crop_prob=None,
        use_global_global=None,
        use_global_local=None,
        use_local_local=None,
        local_conv_layer=None,
        share_seq_mlps=False,
        batch_B=None,
        batch_T=None,
        warmup_T=None,
        contrast_mode=None,
        rnn_size=None,
        bidirectional=None,
        inverse_loss_coeff=None,
        inverse_ent_coeff=None,
        inv_subtract=None,
        inv_use_input=None,
        inv_hidden_sizes=None,
        symmetrize=False,  # BYOL
        pixel_control_coeff=None,
        downsample_rate_T=None,
        downsample_skip_B=None,
        final_conv_nonlinearity=None,
        intensity_sigma=None,
        intensity_prob=None,
        B_sample_mode=None,
    )
)

configs["serial_ddqn"] = config


config = copy.deepcopy(config)

config["algo"]["batch_size"] = 256
config["algo"]["target_update_interval"] = 1000
config["algo"]["learning_rate"] = 1.5e-4
config["optim"]["eps"] = 0.01 / 256
config["sampler"]["batch_T"] = 2
config["sampler"]["batch_B"] = 16
config["sampler"]["eval_n_envs"] = 4
configs["scaled_ddqn"] = config


config = copy.deepcopy(configs["scaled_ddqn"])

config["agent"]["eps_final"] = 0.2
config["agent"]["eps_final_min"] = 0.001
configs["vec_eps"] = config


config = copy.deepcopy(configs["scaled_ddqn"])
config["model"] = dict(
    channels4=0,
    use_maxpool=False,
    skip_connections=False,
    hidden_sizes=512,
    kiaming_init=True,
    stop_grad=None,
)
config["agent"]["load_fc1"] = False
configs["scaled_ddqn_mediumnet"] = config

config = copy.deepcopy(configs["scaled_ddqn"])
config["model"] = dict(
    channels4=64,
    use_maxpool=True,
    skip_connections=True,
    hidden_sizes=512,
    kiaming_init=True,
    stop_grad=None,
)
config["agent"]["load_fc1"] = False
configs["scaled_ddqn_largenet"] = config
