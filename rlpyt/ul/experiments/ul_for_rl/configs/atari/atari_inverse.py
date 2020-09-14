
import copy

configs = dict()

config = dict(
    algo=dict(
        replay_filepath=None,
        batch_size=256,
        delta_T=3,
        learning_rate=1e-3,
        learning_rate_anneal="cosine",  # cosine
        learning_rate_warmup=1000,  # number of updates
        clip_grad_norm=10.,
        random_shift_prob=0.1,
        random_shift_pad=4,
        entropy_loss_coeff=0.01,
        activation_loss_coefficient=0.,  # rarely if ever use
        validation_split=0.0,
        n_validation_batches=0,  # usually don't do it.
    ),
    encoder=dict(
        channels=None,
        kernel_sizes=None,
        strides=None,
        paddings=None,
        hidden_sizes=None,
        kiaming_init=True,
    ),
    inverse_model=dict(
        hidden_sizes=512,
        subtract=False,
    ),
    optim=dict(
        weight_decay=0,
    ),
    runner=dict(
        n_updates=int(3e4),  # 30k Usually sufficient for one game?
        log_interval_updates=int(1e3),
    ),
    name="atari_inv",  # probably change this with the filepath
)

configs["atari_inv"] = config
