
import copy

configs = dict()

config = dict(
    algo=dict(
        replay_filepath=None,
        batch_size=128,
        learning_rate=1e-3,
        delta_T=0,
        learning_rate_anneal="cosine",  # cosine
        learning_rate_warmup=1000,  # number of updates
        clip_grad_norm=10.,
        latent_size=256,
        hidden_sizes=None,
        activation_loss_coefficient=0.,  # rarely if ever use
        kl_coeff=1.,
        onehot_action=True,
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
    decoder=dict(
        reshape=[64, 7, 7],
        channels=[64, 32, 4],
        kernel_sizes=[3, 4, 8],
        strides=[1, 2, 4],
        paddings=[0, 0, 0],
        output_paddings=[0, 0, 0],
    ),
    optim=dict(
        weight_decay=0,
    ),
    runner=dict(
        n_updates=int(2e4),  # 20k Usually sufficient for one game?
        log_interval_updates=int(1e3),
    ),
    name="atari_vae",  # probably change this with the filepath
)

configs["atari_vae"] = config
