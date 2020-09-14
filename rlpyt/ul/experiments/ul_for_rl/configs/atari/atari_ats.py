
import copy

configs = dict()

config = dict(
    algo=dict(
        replay_filepath=None,
        delta_T=3,
        batch_T=1,
        batch_B=512,
        learning_rate=1e-3,
        learning_rate_anneal="cosine",  # cosine
        learning_rate_warmup=1000,  # number of updates
        clip_grad_norm=10.,
        target_update_tau=0.01,   # 1 for hard update
        target_update_interval=1,
        latent_size=256,
        anchor_hidden_sizes=512,
        random_shift_prob=0.1,
        random_shift_pad=4,
        activation_loss_coefficient=0.,  # rarely if ever use
        validation_split=0.0,
        n_validation_batches=0,  # usually don't do it.
    ),
    encoder=dict(
        channels=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[4, 2, 1],
        paddings=None,
        hidden_sizes=None,
        kiaming_init=True,
    ),
    optim=dict(
        weight_decay=0,
    ),
    runner=dict(
        n_updates=int(2e4),  # 20k Usually sufficient for one game?
        log_interval_updates=int(1e3),
    ),
    name="atari_ats",  # probably change this with the filepath
)

configs["atari_ats"] = config
